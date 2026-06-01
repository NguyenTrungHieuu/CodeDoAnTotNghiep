%%writefile ddp_distilbert_nrms_baseline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_auc_score, ndcg_score

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ====================== DISTRIBUTED SETUP ======================
def init_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

init_distributed()
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
rank = dist.get_rank()

# ====================== DATA PATH (chuẩn paper) ======================
BASE = "/kaggle/input/datasets/nguyenueih12211221/dataset-mind-small"
news_path = f"{BASE}/MINDsmall_train/news.tsv"
behaviors_train_path = f"{BASE}/MINDsmall_train/behaviors.tsv"
behaviors_val_path = f"{BASE}/MINDsmall_val/behaviors.tsv"

NEWS_COLUMNS = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
BEHAVIOR_COLUMNS = ['impression_id', 'user_id', 'time', 'history', 'impressions']

news_df = pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS)
behaviors_train_df = pd.read_csv(behaviors_train_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
behaviors_val_df = pd.read_csv(behaviors_val_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)

behaviors_test_df = behaviors_val_df.copy()   # Test = Val (chuẩn paper)

news_df['title'] = news_df['title'].fillna('')
news_df['text'] = news_df['title']

for df in [behaviors_train_df, behaviors_val_df, behaviors_test_df]:
    df['history'] = df['history'].fillna('')
    df['impressions'] = df['impressions'].fillna('')

if rank == 0:
    print(f"News: {len(news_df):,} | Train behaviors: {len(behaviors_train_df):,} | Val/Test behaviors: {len(behaviors_val_df):,}")

# ====================== DISTILBERT TITLE EMBEDDING ======================
embedding_path = "/kaggle/working/distilbert_title_embeddings.npy"
news_ids_path = "/kaggle/working/distilbert_news_ids.npy"

if rank == 0:
    print("Rank 0: Encoding titles with DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    distilbert.eval()

    news_id_list = news_df['news_id'].tolist()
    title_texts = news_df['text'].tolist()
    batch_size = 128
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(title_texts), batch_size), desc="Encoding titles"):
            batch = title_texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = distilbert(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
            embeddings.append(batch_emb)
    embeddings = np.concatenate(embeddings, axis=0)
    np.save(embedding_path, embeddings)
    np.save(news_ids_path, np.array(news_id_list))

dist.barrier()
embeddings = np.load(embedding_path)
news_id_list = np.load(news_ids_path).tolist()
news_dict = {nid: embeddings[i] for i, nid in enumerate(news_id_list)}

if rank == 0:
    print(f"Loaded {len(news_dict):,} news embeddings.")

# ====================== DATASET (K=3, max_history=60 theo paper) ======================
class ViNewsRecDataset(Dataset):
    def __init__(self, behaviors, news_dict, k=3, max_history=60):
        self.behaviors = behaviors
        self.news_dict = news_dict
        self.k = k
        self.max_history = max_history
        self.history_list = []
        self.candidates_list = []
        self.labels_list = []
        zero_emb = np.zeros(768, dtype=np.float32)

        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), disable=(rank != 0)):
            # History
            history_ids = row['history'].split()[-self.max_history:]
            h = [self.news_dict.get(nid, zero_emb) for nid in history_ids]
            if not h:
                h = [zero_emb]

            # Positive & Negative
            pos, neg = [], []
            for imp in row['impressions'].split():
                if '-' not in imp: continue
                nid, label = imp.split('-')
                if nid in self.news_dict:
                    if label == '1':
                        pos.append(self.news_dict[nid])
                    else:
                        neg.append(self.news_dict[nid])

            if not pos: continue
            pos_emb = pos[np.random.randint(len(pos))]

            # Negative sampling K=3
            if len(neg) == 0:
                negs = [zero_emb] * self.k
            else:
                neg_indices = np.random.choice(len(neg), self.k, replace=True)
                negs = [neg[i] for i in neg_indices]

            candidates = [np.array(negs[i]) for i in range(self.k)]
            pos_position = np.random.randint(0, self.k + 1)
            candidates.insert(pos_position, pos_emb)

            self.history_list.append(h)
            self.candidates_list.append(candidates)
            self.labels_list.append(pos_position)

    def __len__(self):
        return len(self.history_list)

    def __getitem__(self, idx):
        history = torch.from_numpy(np.array(self.history_list[idx])).float()
        candidates = torch.from_numpy(np.array(self.candidates_list[idx])).float()
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)
        return history, candidates, label

    @staticmethod
    def collate_fn(batch):
        history = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0.0)
        candidates = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return history, candidates, labels

# ====================== NRMS MODEL ======================
class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        e = torch.tanh(self.fc1(x))
        alpha = self.fc2(e).squeeze(-1)
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(alpha, dim=1)
        return torch.sum(x * alpha.unsqueeze(-1), dim=1)

class NewsEncoder(nn.Module):
    def __init__(self, embed_size=768, encoder_size=256):
        super().__init__()
        self.proj = nn.Linear(embed_size, encoder_size)

    def forward(self, x):
        return F.dropout(self.proj(x), p=0.3, training=self.training)

class UserEncoder(nn.Module):
    def __init__(self, encoder_size=256, nhead=16, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(encoder_size, nhead, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(encoder_size, encoder_size)
        self.att = AdditiveAttention(encoder_size)

    def forward(self, x):
        pad_mask = (x.sum(dim=-1) == 0)
        x, _ = self.mha(x, x, x, key_padding_mask=pad_mask)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.proj(x)
        return self.att(x, mask=(~pad_mask).float())

class NRMS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.news_encoder = NewsEncoder()
        self.user_encoder = UserEncoder()
        self.device = device

    def forward(self, history, candidates, labels=None):
        h_emb = self.news_encoder(history)
        c_emb = self.news_encoder(candidates)
        user = self.user_encoder(h_emb)
        scores = torch.bmm(user.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)
        if labels is not None:
            loss = F.cross_entropy(scores, labels.to(self.device))
            return loss, scores
        return scores

# ====================== EVALUATION (đúng chuẩn paper) ======================
def full_impression_eval(model, behaviors_df, news_dict, device, max_history=60):
    model.eval()
    all_scores = []
    all_y_true = []
    zero_emb = np.zeros(768, dtype=np.float32)

    with torch.no_grad():
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), disable=(rank != 0)):
            history_ids = row['history'].split()[-max_history:] if row['history'] else []
            history_emb = [news_dict.get(nid, zero_emb) for nid in history_ids]
            if not history_emb:
                history_emb = [zero_emb]
            history_tensor = torch.from_numpy(np.array(history_emb)).float().unsqueeze(0).to(device)

            impressions = row['impressions'].split() if row['impressions'] else []
            candidate_ids, labels = [], []
            for imp in impressions:
                if '-' not in imp: continue
                nid, label = imp.split('-')
                if nid in news_dict:
                    candidate_ids.append(nid)
                    labels.append(int(label))

            if len(candidate_ids) < 2 or sum(labels) == 0:
                continue

            candidate_tensor = torch.from_numpy(
                np.array([news_dict[nid] for nid in candidate_ids])
            ).float().unsqueeze(0).to(device)

            h_emb = model.news_encoder(history_tensor)
            user = model.user_encoder(h_emb)
            c_emb = model.news_encoder(candidate_tensor)
            scores = torch.matmul(user, c_emb.transpose(1, 2)).squeeze(1).cpu().numpy()[0]

            all_scores.append(scores)
            all_y_true.append(np.array(labels))

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for scores, y_true in zip(all_scores, all_y_true):
        if y_true.sum() == 0: continue
        aucs.append(roc_auc_score(y_true, scores))
        sorted_idx = np.argsort(-scores)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(scores) + 1)
        pos_ranks = ranks[y_true == 1]
        if len(pos_ranks) > 0:
            mrrs.append(1.0 / pos_ranks.min())

        if len(scores) >= 2:
            ndcg5s.append(ndcg_score([y_true], [scores], k=5))
            ndcg10s.append(ndcg_score([y_true], [scores], k=10))

    return (np.mean(aucs) if aucs else 0.0,
            np.mean(mrrs) if mrrs else 0.0,
            np.mean(ndcg5s) if ndcg5s else 0.0,
            np.mean(ndcg10s) if ndcg10s else 0.0)

# ====================== TRAINING ======================
train_dataset = ViNewsRecDataset(behaviors_train_df, news_dict, k=3, max_history=60)
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler,
                          collate_fn=ViNewsRecDataset.collate_fn, pin_memory=True, num_workers=4)

model = NRMS(device).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank])

if rank == 0:
    total_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print(f"\n=== MODEL SUMMARY ===\nTotal trainable parameters: {total_params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5
best_auc = 0.0
best_model_path = "/kaggle/working/best_nrms_distilbert.pth"

for epoch in range(epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0)):
        history, candidates, labels = [x.to(device) for x in batch]
        loss, _ = model(history, candidates, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    if rank == 0:
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = full_impression_eval(
            model.module, behaviors_val_df, news_dict, device)
        print(f"Val  | AUC: {val_auc:.4f} | MRR: {val_mrr:.4f} | NDCG@5: {val_ndcg5:.4f} | NDCG@10: {val_ndcg10:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), best_model_path)
            print(">>> Saved best model!")

    dist.barrier()

# ====================== FINAL TEST ======================
if rank == 0:
    print("\n=== FINAL TEST EVALUATION (same as Val) ===")
    test_model = NRMS(device).to(device)
    test_model.load_state_dict(torch.load(best_model_path))
    test_auc, test_mrr, test_ndcg5, test_ndcg10 = full_impression_eval(
        test_model, behaviors_test_df, news_dict, device)
    print(f"Test | AUC: {test_auc:.4f} | MRR: {test_mrr:.4f} | NDCG@5: {test_ndcg5:.4f} | NDCG@10: {test_ndcg10:.4f}")

dist.barrier()
dist.destroy_process_group()

!torchrun --nproc_per_node=2 ddp_distilbert_nrms_baseline.py

%%writefile ddp_distilbert_nrms_baseline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import random
from sklearn.metrics import roc_auc_score, ndcg_score

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ====================== DISTRIBUTED SETUP ======================
def init_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = init_distributed()
device = torch.device("cuda", local_rank)
rank = dist.get_rank()

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if rank == 0:
    print("✅ Seeds set for reproducibility")

# ====================== DATA PATH ======================
BASE = "/kaggle/input/datasets/nguyenueih12211221/dataset-mind-small"
news_path = f"{BASE}/MINDsmall_train/news.tsv"
behaviors_train_path = f"{BASE}/MINDsmall_train/behaviors.tsv"
behaviors_val_path = f"{BASE}/MINDsmall_val/behaviors.tsv"

NEWS_COLUMNS = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
BEHAVIOR_COLUMNS = ['impression_id', 'user_id', 'time', 'history', 'impressions']

news_df = pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS)
behaviors_train_df = pd.read_csv(behaviors_train_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
behaviors_val_df = pd.read_csv(behaviors_val_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
behaviors_test_df = behaviors_val_df.copy()  # Test = Val (theo paper)

news_df['title'] = news_df['title'].fillna('')
news_df['text'] = news_df['title']
for df in [behaviors_train_df, behaviors_val_df, behaviors_test_df]:
    df['history'] = df['history'].fillna('')
    df['impressions'] = df['impressions'].fillna('')

if rank == 0:
    print(f"News: {len(news_df):,} | Train behaviors: {len(behaviors_train_df):,} | Val/Test: {len(behaviors_val_df):,}")

# ====================== DISTILBERT TITLE EMBEDDING (CACHE) ======================
embedding_path = "/kaggle/working/distilbert_title_embeddings.npy"
news_ids_path = "/kaggle/working/distilbert_news_ids.npy"

if rank == 0:
    if os.path.exists(embedding_path) and os.path.exists(news_ids_path):
        print("✅ Rank 0: Using cached DistilBERT title embeddings")
    else:
        print("🔄 Rank 0: Encoding titles with DistilBERT...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        distilbert = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        distilbert.eval()

        news_id_list = news_df['news_id'].tolist()
        title_texts = news_df['text'].tolist()
        batch_size = 128
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(title_texts), batch_size), desc="Encoding titles"):
                batch = title_texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                outputs = distilbert(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
                embeddings.append(batch_emb)
        embeddings = np.concatenate(embeddings, axis=0)
        np.save(embedding_path, embeddings)
        np.save(news_ids_path, np.array(news_id_list))

dist.barrier()

embeddings = np.load(embedding_path)
news_id_list = np.load(news_ids_path).tolist()
news_dict = {nid: embeddings[i] for i, nid in enumerate(news_id_list)}
if rank == 0:
    print(f"Loaded {len(news_dict):,} news embeddings.")

# ====================== DATASET ======================
class ViNewsRecDataset(Dataset):
    def __init__(self, behaviors, news_dict, k=3, max_history=60):
        self.behaviors = behaviors
        self.news_dict = news_dict
        self.k = k
        self.max_history = max_history
        self.history_list = []
        self.candidates_list = []
        self.labels_list = []
        zero_emb = np.zeros(768, dtype=np.float32)

        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), disable=(rank != 0)):
            # History (most recent 60)
            history_ids = row['history'].split()[-self.max_history:]
            h = [self.news_dict.get(nid, zero_emb) for nid in history_ids]
            if not h:
                h = [zero_emb]

            # Positive & Negative
            pos, neg = [], []
            for imp in row['impressions'].split():
                if '-' not in imp:
                    continue
                nid, label = imp.split('-')
                if nid in self.news_dict:
                    if label == '1':
                        pos.append(self.news_dict[nid])
                    else:
                        neg.append(self.news_dict[nid])

            if not pos:
                continue
            pos_emb = pos[np.random.randint(len(pos))]

            # Negative sampling K=3
            if len(neg) == 0:
                negs = [zero_emb] * self.k
            else:
                neg_indices = np.random.choice(len(neg), self.k, replace=True)
                negs = [neg[i] for i in neg_indices]

            candidates = [np.array(negs[i]) for i in range(self.k)]
            pos_position = np.random.randint(0, self.k + 1)
            candidates.insert(pos_position, pos_emb)

            self.history_list.append(h)
            self.candidates_list.append(candidates)
            self.labels_list.append(pos_position)

    def __len__(self):
        return len(self.history_list)

    def __getitem__(self, idx):
        history = torch.from_numpy(np.array(self.history_list[idx])).float()
        candidates = torch.from_numpy(np.array(self.candidates_list[idx])).float()
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)
        return history, candidates, label

    @staticmethod
    def collate_fn(batch):
        history = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0.0)
        candidates = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return history, candidates, labels

# ====================== NRMS MODEL (theo paper) ======================
class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        e = torch.tanh(self.fc1(x))
        alpha = self.fc2(e).squeeze(-1)
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(alpha, dim=1)
        return torch.sum(x * alpha.unsqueeze(-1), dim=1)

class NewsEncoder(nn.Module):
    def __init__(self, embed_size=768, encoder_size=256):
        super().__init__()
        self.proj = nn.Linear(embed_size, encoder_size)

    def forward(self, x):
        return F.dropout(self.proj(x), p=0.3, training=self.training)

class UserEncoder(nn.Module):
    def __init__(self, encoder_size=256, nhead=16, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(encoder_size, nhead, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(encoder_size, encoder_size)
        self.att = AdditiveAttention(encoder_size)

    def forward(self, x):
        pad_mask = (x.sum(dim=-1) == 0)          # True = padding
        x, _ = self.mha(x, x, x, key_padding_mask=pad_mask)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.proj(x)
        return self.att(x, mask=(~pad_mask).float())

class NRMS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.news_encoder = NewsEncoder()
        self.user_encoder = UserEncoder()
        self.device = device

    def forward(self, history, candidates, labels=None):
        h_emb = self.news_encoder(history)      # [B, seq, 256]
        c_emb = self.news_encoder(candidates)   # [B, num_cand, 256]
        user = self.user_encoder(h_emb)         # [B, 256]
        scores = torch.bmm(user.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)  # [B, num_cand]

        if labels is not None:
            loss = F.cross_entropy(scores, labels.to(self.device))
            return loss, scores
        return scores

# ====================== EVALUATION (chuẩn paper) ======================
def full_impression_eval(model, behaviors_df, news_dict, device, max_history=60):
    model.eval()
    all_scores = []
    all_y_true = []
    zero_emb = np.zeros(768, dtype=np.float32)

    with torch.no_grad():
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), disable=(rank != 0)):
            # History
            history_ids = row['history'].split()[-max_history:] if row['history'] else []
            history_emb = [news_dict.get(nid, zero_emb) for nid in history_ids]
            if not history_emb:
                history_emb = [zero_emb]
            history_tensor = torch.from_numpy(np.array(history_emb)).float().unsqueeze(0).to(device)

            # Candidates
            impressions = row['impressions'].split() if row['impressions'] else []
            candidate_ids, labels = [], []
            for imp in impressions:
                if '-' not in imp: continue
                nid, label = imp.split('-')
                if nid in news_dict:
                    candidate_ids.append(nid)
                    labels.append(int(label))

            if len(candidate_ids) < 2 or sum(labels) == 0:
                continue

            candidate_tensor = torch.from_numpy(
                np.array([news_dict[nid] for nid in candidate_ids])
            ).float().unsqueeze(0).to(device)

            h_emb = model.news_encoder(history_tensor)
            user = model.user_encoder(h_emb)
            c_emb = model.news_encoder(candidate_tensor)
            scores = torch.matmul(user, c_emb.transpose(1, 2)).squeeze(1).cpu().numpy()[0]

            all_scores.append(scores)
            all_y_true.append(np.array(labels))

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for scores, y_true in zip(all_scores, all_y_true):
        if y_true.sum() == 0: continue
        aucs.append(roc_auc_score(y_true, scores))
        sorted_idx = np.argsort(-scores)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(scores) + 1)
        pos_ranks = ranks[y_true == 1]
        if len(pos_ranks) > 0:
            mrrs.append(1.0 / pos_ranks.min())
        if len(scores) >= 2:
            ndcg5s.append(ndcg_score([y_true], [scores], k=5))
            ndcg10s.append(ndcg_score([y_true], [scores], k=10))

    return (np.mean(aucs) if aucs else 0.0,
            np.mean(mrrs) if mrrs else 0.0,
            np.mean(ndcg5s) if ndcg5s else 0.0,
            np.mean(ndcg10s) if ndcg10s else 0.0)

# ====================== TRAINING ======================
train_dataset = ViNewsRecDataset(behaviors_train_df, news_dict, k=3, max_history=60)
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler,
                          collate_fn=ViNewsRecDataset.collate_fn, pin_memory=True, num_workers=4)

model = NRMS(device).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank])

if rank == 0:
    total_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print(f"\n=== MODEL SUMMARY ===\nTotal trainable parameters: {total_params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5
best_auc = 0.0
best_model_path = "/kaggle/working/best_nrms_distilbert.pth"

for epoch in range(epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0)):
        history, candidates, labels = [x.to(device) for x in batch]
        loss, _ = model(history, candidates, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # Validation chỉ rank 0
    if rank == 0:
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = full_impression_eval(
            model.module, behaviors_val_df, news_dict, device)
        print(f"Val | AUC: {val_auc:.4f} | MRR: {val_mrr:.4f} | NDCG@5: {val_ndcg5:.4f} | NDCG@10: {val_ndcg10:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), best_model_path)
            print(">>> Saved best model!")

    dist.barrier()

# ====================== FINAL TEST ======================
if rank == 0:
    print("\n=== FINAL TEST EVALUATION ===")
    test_model = NRMS(device).to(device)
    test_model.load_state_dict(torch.load(best_model_path))
    test_auc, test_mrr, test_ndcg5, test_ndcg10 = full_impression_eval(
        test_model, behaviors_test_df, news_dict, device)
    print(f"Test | AUC: {test_auc:.4f} | MRR: {test_mrr:.4f} | NDCG@5: {test_ndcg5:.4f} | NDCG@10: {test_ndcg10:.4f}")

dist.barrier()
dist.destroy_process_group()

!torchrun --nproc_per_node=2 ddp_distilbert_nrms_baseline.py

%%writefile ddp_glove_nrms_baseline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import random
import subprocess
from sklearn.metrics import roc_auc_score, ndcg_score

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== DISTRIBUTED SETUP ======================
def init_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = init_distributed()
device = torch.device("cuda", local_rank)
rank = dist.get_rank()

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if rank == 0:
    print("✅ Seeds set for reproducibility")

# ====================== DATA PATH ======================
BASE = "/kaggle/input/datasets/nguyenueih12211221/dataset-mind-small"
news_path = f"{BASE}/MINDsmall_train/news.tsv"
behaviors_train_path = f"{BASE}/MINDsmall_train/behaviors.tsv"
behaviors_val_path = f"{BASE}/MINDsmall_val/behaviors.tsv"

NEWS_COLUMNS = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
BEHAVIOR_COLUMNS = ['impression_id', 'user_id', 'time', 'history', 'impressions']

news_df = pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS)
behaviors_train_df = pd.read_csv(behaviors_train_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
behaviors_val_df = pd.read_csv(behaviors_val_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
behaviors_test_df = behaviors_val_df.copy()

news_df['title'] = news_df['title'].fillna('')
news_df['text'] = news_df['title']
for df in [behaviors_train_df, behaviors_val_df, behaviors_test_df]:
    df['history'] = df['history'].fillna('')
    df['impressions'] = df['impressions'].fillna('')

if rank == 0:
    print(f"News: {len(news_df):,} | Train behaviors: {len(behaviors_train_df):,} | Val/Test: {len(behaviors_val_df):,}")

# ====================== GLOVE 300d STATIC EMBEDDINGS (CHUẨN PAPER) ======================
embedding_path = "/kaggle/working/glove_title_embeddings.npy"
news_ids_path = "/kaggle/working/glove_news_ids.npy"

if rank == 0:
    if os.path.exists(embedding_path) and os.path.exists(news_ids_path):
        print("✅ Using cached GloVe title embeddings")
    else:
        print("🔄 Downloading and processing GloVe 300d...")
        
        # Download GloVe if not exists
        glove_zip = "/kaggle/working/glove.6B.zip"
        glove_txt = "/kaggle/working/glove.6B.300d.txt"
        if not os.path.exists(glove_txt):
            if not os.path.exists(glove_zip):
                subprocess.call(["wget", "http://nlp.stanford.edu/data/glove.6B.zip", "-O", glove_zip])
            subprocess.call(["unzip", "-q", glove_zip, "-d", "/kaggle/working/"])
        
        # Load GloVe
        glove_dict = {}
        with open(glove_txt, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe 300d"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_dict[word] = vector
        
        # Compute mean-pooling embedding cho mỗi title
        embeddings = []
        news_id_list = news_df['news_id'].tolist()
        for text in tqdm(news_df['text'], desc="Computing title embeddings (GloVe)"):
            words = text.lower().split()
            vecs = [glove_dict.get(w, np.zeros(300, dtype=np.float32)) for w in words]
            emb = np.mean(vecs, axis=0) if vecs else np.zeros(300, dtype=np.float32)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        np.save(embedding_path, embeddings)
        np.save(news_ids_path, np.array(news_id_list))
        print(f"Saved GloVe embeddings for {len(embeddings):,} news")

dist.barrier()

embeddings = np.load(embedding_path)
news_id_list = np.load(news_ids_path).tolist()
news_dict = {nid: embeddings[i] for i, nid in enumerate(news_id_list)}
if rank == 0:
    print(f"Loaded {len(news_dict):,} news embeddings (GloVe 300d)")

# ====================== DATASET ======================
class ViNewsRecDataset(Dataset):
    def __init__(self, behaviors, news_dict, k=3, max_history=60):
        self.behaviors = behaviors
        self.news_dict = news_dict
        self.k = k
        self.max_history = max_history
        self.history_list = []
        self.candidates_list = []
        self.labels_list = []
        zero_emb = np.zeros(300, dtype=np.float32)   # ← GloVe 300d

        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), disable=(rank != 0)):
            history_ids = row['history'].split()[-self.max_history:]
            h = [self.news_dict.get(nid, zero_emb) for nid in history_ids]
            if not h:
                h = [zero_emb]

            pos, neg = [], []
            for imp in row['impressions'].split():
                if '-' not in imp: continue
                nid, label = imp.split('-')
                if nid in self.news_dict:
                    if label == '1':
                        pos.append(self.news_dict[nid])
                    else:
                        neg.append(self.news_dict[nid])

            if not pos: continue
            pos_emb = pos[np.random.randint(len(pos))]

            if len(neg) == 0:
                negs = [zero_emb] * self.k
            else:
                neg_indices = np.random.choice(len(neg), self.k, replace=True)
                negs = [neg[i] for i in neg_indices]

            candidates = [np.array(negs[i]) for i in range(self.k)]
            pos_position = np.random.randint(0, self.k + 1)
            candidates.insert(pos_position, pos_emb)

            self.history_list.append(h)
            self.candidates_list.append(candidates)
            self.labels_list.append(pos_position)

    def __len__(self):
        return len(self.history_list)

    def __getitem__(self, idx):
        history = torch.from_numpy(np.array(self.history_list[idx])).float()
        candidates = torch.from_numpy(np.array(self.candidates_list[idx])).float()
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)
        return history, candidates, label

    @staticmethod
    def collate_fn(batch):
        history = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0.0)
        candidates = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return history, candidates, labels

# ====================== NRMS MODEL (giữ nguyên cấu trúc paper) ======================
class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        e = torch.tanh(self.fc1(x))
        alpha = self.fc2(e).squeeze(-1)
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(alpha, dim=1)
        return torch.sum(x * alpha.unsqueeze(-1), dim=1)

class NewsEncoder(nn.Module):
    def __init__(self, embed_size=300, encoder_size=256):   # ← 300d GloVe
        super().__init__()
        self.proj = nn.Linear(embed_size, encoder_size)

    def forward(self, x):
        return F.dropout(self.proj(x), p=0.3, training=self.training)

class UserEncoder(nn.Module):
    def __init__(self, encoder_size=256, nhead=16, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(encoder_size, nhead, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(encoder_size, encoder_size)
        self.att = AdditiveAttention(encoder_size)

    def forward(self, x):
        pad_mask = (x.sum(dim=-1) == 0)
        x, _ = self.mha(x, x, x, key_padding_mask=pad_mask)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.proj(x)
        return self.att(x, mask=(~pad_mask).float())

class NRMS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.news_encoder = NewsEncoder()
        self.user_encoder = UserEncoder()
        self.device = device

    def forward(self, history, candidates, labels=None):
        h_emb = self.news_encoder(history)
        c_emb = self.news_encoder(candidates)
        user = self.user_encoder(h_emb)
        scores = torch.bmm(user.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)
        if labels is not None:
            loss = F.cross_entropy(scores, labels.to(self.device))
            return loss, scores
        return scores

# ====================== EVALUATION ======================
def full_impression_eval(model, behaviors_df, news_dict, device, max_history=60):
    model.eval()
    all_scores = []
    all_y_true = []
    zero_emb = np.zeros(300, dtype=np.float32)   # ← GloVe 300d

    with torch.no_grad():
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), disable=(rank != 0)):
            history_ids = row['history'].split()[-max_history:] if row['history'] else []
            history_emb = [news_dict.get(nid, zero_emb) for nid in history_ids]
            if not history_emb:
                history_emb = [zero_emb]
            history_tensor = torch.from_numpy(np.array(history_emb)).float().unsqueeze(0).to(device)

            impressions = row['impressions'].split() if row['impressions'] else []
            candidate_ids, labels = [], []
            for imp in impressions:
                if '-' not in imp: continue
                nid, label = imp.split('-')
                if nid in news_dict:
                    candidate_ids.append(nid)
                    labels.append(int(label))

            if len(candidate_ids) < 2 or sum(labels) == 0:
                continue

            candidate_tensor = torch.from_numpy(
                np.array([news_dict[nid] for nid in candidate_ids])
            ).float().unsqueeze(0).to(device)

            h_emb = model.news_encoder(history_tensor)
            user = model.user_encoder(h_emb)
            c_emb = model.news_encoder(candidate_tensor)
            scores = torch.matmul(user, c_emb.transpose(1, 2)).squeeze(1).cpu().numpy()[0]

            all_scores.append(scores)
            all_y_true.append(np.array(labels))

    # Tính metric giống paper
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for scores, y_true in zip(all_scores, all_y_true):
        if y_true.sum() == 0: continue
        aucs.append(roc_auc_score(y_true, scores))
        sorted_idx = np.argsort(-scores)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(scores) + 1)
        pos_ranks = ranks[y_true == 1]
        if len(pos_ranks) > 0:
            mrrs.append(1.0 / pos_ranks.min())
        if len(scores) >= 2:
            ndcg5s.append(ndcg_score([y_true], [scores], k=5))
            ndcg10s.append(ndcg_score([y_true], [scores], k=10))

    return (np.mean(aucs) if aucs else 0.0,
            np.mean(mrrs) if mrrs else 0.0,
            np.mean(ndcg5s) if ndcg5s else 0.0,
            np.mean(ndcg10s) if ndcg10s else 0.0)

# ====================== TRAINING ======================
train_dataset = ViNewsRecDataset(behaviors_train_df, news_dict, k=3, max_history=60)
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler,
                          collate_fn=ViNewsRecDataset.collate_fn, pin_memory=True, num_workers=4)

model = NRMS(device).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank])

if rank == 0:
    total_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print(f"\n=== MODEL SUMMARY (NRMS + GloVe 300d) ===\nTotal trainable parameters: {total_params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5
best_auc = 0.0
best_model_path = "/kaggle/working/best_nrms_glove.pth"

for epoch in range(epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0)):
        history, candidates, labels = [x.to(device) for x in batch]
        loss, _ = model(history, candidates, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    if rank == 0:
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = full_impression_eval(
            model.module, behaviors_val_df, news_dict, device)
        print(f"Val | AUC: {val_auc:.4f} | MRR: {val_mrr:.4f} | NDCG@5: {val_ndcg5:.4f} | NDCG@10: {val_ndcg10:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), best_model_path)
            print(">>> Saved best model!")

    dist.barrier()

# ====================== FINAL TEST ======================
if rank == 0:
    print("\n=== FINAL TEST EVALUATION ===")
    test_model = NRMS(device).to(device)
    test_model.load_state_dict(torch.load(best_model_path))
    test_auc, test_mrr, test_ndcg5, test_ndcg10 = full_impression_eval(
        test_model, behaviors_test_df, news_dict, device)
    print(f"Test | AUC: {test_auc:.4f} | MRR: {test_mrr:.4f} | NDCG@5: {test_ndcg5:.4f} | NDCG@10: {test_ndcg10:.4f}")

dist.barrier()
dist.destroy_process_group()

!torchrun --nproc_per_node=2 ddp_glove_nrms_baseline.py