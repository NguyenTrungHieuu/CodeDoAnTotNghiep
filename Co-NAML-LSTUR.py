# ============================================================
# INSTALL & IMPORT
# ============================================================
!pip install transformers accelerate --quiet

import os, math, pickle, random, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# CONFIGURATION
# ============================================================
config = {
    "num_filters": 300,
    "query_vector_dim": 200,
    "dropout_probability": 0.2,
    "window_size": 3,
    "num_categories": None,
    "category_embedding_dim": 100,
    "num_users": None,
    "masking_probability": 0.5,
    "max_history": 50,
    "max_title_len": 30,
    "max_abstract_len": 100,
    "negative_sampling_ratio": 4,
    "batch_size": 64,
    "epochs": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dataset_attributes": {"news": ["category", "subcategory", "title", "abstract"]}
}

# Đường dẫn MIND-small
TRAIN_NEWS_PATH = "/kaggle/input/datasets/neyugn12/dataset-mind-small/MINDsmall_train/news.tsv"
VAL_NEWS_PATH   = "/kaggle/input/datasets/neyugn12/dataset-mind-small/MINDsmall_val/news.tsv"
TRAIN_BEHAVIOR_PATH = "/kaggle/input/datasets/neyugn12/dataset-mind-small/MINDsmall_train/behaviors.tsv"
VAL_BEHAVIOR_PATH   = "/kaggle/input/datasets/neyugn12/dataset-mind-small/MINDsmall_val/behaviors.tsv"

# ============================================================
# LOAD & PROCESS NEWS DATA
# ============================================================
news_columns = ["news_id", "category", "subcategory", "title", "abstract",
                "url", "title_entities", "abstract_entities"]
train_news = pd.read_csv(TRAIN_NEWS_PATH, sep="\t", header=None, names=news_columns)
val_news   = pd.read_csv(VAL_NEWS_PATH,   sep="\t", header=None, names=news_columns)

cols_keep = ["news_id", "category", "subcategory", "title", "abstract"]
train_news = train_news[cols_keep]
val_news   = val_news[cols_keep]

all_news = pd.concat([train_news, val_news], ignore_index=True).fillna("")

cat_encoder = LabelEncoder()
subcat_encoder = LabelEncoder()
cat_encoder.fit(all_news["category"].unique())
subcat_encoder.fit(all_news["subcategory"].unique())

all_news["cat_id"] = cat_encoder.transform(all_news["category"]) + 1
all_news["subcat_id"] = subcat_encoder.transform(all_news["subcategory"]) + 1

config["num_categories"] = max(all_news["cat_id"].max(), all_news["subcat_id"].max()) + 1
print(f"News: {len(all_news)}, Categories: {config['num_categories']}")

# ============================================================
# LOAD BEHAVIORS & USER MAPPING
# ============================================================
behaviors_columns = ["impression_id", "user_id", "time", "history", "impressions"]
behaviors_train = pd.read_csv(TRAIN_BEHAVIOR_PATH, sep="\t", header=None, names=behaviors_columns)
behaviors_val   = pd.read_csv(VAL_BEHAVIOR_PATH,   sep="\t", header=None, names=behaviors_columns)

users = set(behaviors_train["user_id"].unique()) | set(behaviors_val["user_id"].unique())
user2idx = {u: i+1 for i, u in enumerate(users)}
config["num_users"] = len(user2idx) + 1
print(f"Users: {config['num_users']}")

# ============================================================
# ADDITIVE ATTENTION MODULE
# ============================================================
class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_vector_dim, bias=False)
        self.query_vector = nn.Parameter(torch.randn(query_vector_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        attn = torch.tanh(self.linear(x))
        weights = F.softmax(torch.matmul(attn, self.query_vector), dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

# ============================================================
# DISTILBERT TEXT ENCODER
# ============================================================
class DistilBertTextEncoder(nn.Module):
    def __init__(self, num_filters, query_vector_dim, dropout_probability, max_length=64):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(num_filters * 2, num_filters)
        )
        self.attention = AdditiveAttention(query_vector_dim, num_filters)
        self.max_length = max_length

    def forward(self, texts):
        encoded = self.tokenizer(
            texts, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attn_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        embeddings = outputs.last_hidden_state
        projected = self.projection(embeddings)
        return self.attention(projected)

# ============================================================
# ELEMENT ENCODER (Category/Subcategory)
# ============================================================
class ElementEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, output_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))

# ============================================================
# NEWS ENCODER
# ============================================================
class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoders = nn.ModuleDict({
            'title': DistilBertTextEncoder(
                config['num_filters'], config['query_vector_dim'],
                config['dropout_probability'], max_length=config['max_title_len']
            ),
            'abstract': DistilBertTextEncoder(
                config['num_filters'], config['query_vector_dim'],
                config['dropout_probability'], max_length=config['max_abstract_len']
            )
        })
        self.element_encoders = nn.ModuleDict({
            'category': ElementEncoder(
                config['num_categories'], config['category_embedding_dim'], config['num_filters']
            ),
            'subcategory': ElementEncoder(
                config['num_categories'], config['category_embedding_dim'], config['num_filters']
            )
        })
        self.final_attention = AdditiveAttention(config['query_vector_dim'], config['num_filters'])

    def forward(self, news):
        vectors = []
        if 'title' in news and news['title'] is not None:
            vectors.append(self.text_encoders['title'](news['title']))
        if 'abstract' in news and news['abstract'] is not None:
            vectors.append(self.text_encoders['abstract'](news['abstract']))
        if 'category' in news and news['category'] is not None:
            vectors.append(self.element_encoders['category'](news['category'].to(device)))
        if 'subcategory' in news and news['subcategory'] is not None:
            vectors.append(self.element_encoders['subcategory'](news['subcategory'].to(device)))
        if len(vectors) == 1:
            return vectors[0]
        return self.final_attention(torch.stack(vectors, dim=1))

# ============================================================
# PRE-ENCODE NEWS VECTORS
# ============================================================
print("\nPre-encoding news vectors...")
news_encoder_temp = NewsEncoder(config).to(device)
news_encoder_temp.eval()
news_vectors = {}

with torch.no_grad():
    for i in tqdm(range(0, len(all_news), 32)):
        batch = all_news.iloc[i:i+32]
        news_batch = {
            'title': batch['title'].tolist(),
            'abstract': batch['abstract'].tolist(),
            'category': torch.tensor(batch['cat_id'].values),
            'subcategory': torch.tensor(batch['subcat_id'].values)
        }
        vectors = news_encoder_temp(news_batch).cpu()
        for nid, vec in zip(batch['news_id'], vectors):
            news_vectors[nid] = vec

print(f"Encoded {len(news_vectors)} articles")
del news_encoder_temp
torch.cuda.empty_cache()

# ============================================================
# DKN ATTENTION MODULE
# ============================================================
class DKNAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(config['num_filters'] * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, candidate_news_vector, clicked_news_vector):
        batch_size = clicked_news_vector.size(0)
        num_clicked = clicked_news_vector.size(1)
        expanded_candidate = candidate_news_vector.unsqueeze(1).expand(-1, num_clicked, -1)
        concat_vectors = torch.cat((expanded_candidate, clicked_news_vector), dim=2)
        attention_scores = self.dnn(concat_vectors).squeeze(dim=2)
        clicked_news_weights = F.softmax(attention_scores, dim=1)
        return torch.bmm(clicked_news_weights.unsqueeze(1), clicked_news_vector).squeeze(1)

# ============================================================
# USER ENCODER (LSTUR-ini)
# ============================================================
class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config['num_filters'] * 1.5)
        self.lstm = nn.LSTM(config['num_filters'], self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, config['num_filters'])

    def forward(self, long_term_emb, clicked_news_length, clicked_news_vector):
        lengths = clicked_news_length.clone()
        lengths[lengths == 0] = 1
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        sorted_vectors = clicked_news_vector[sorted_idx]
        sorted_long_term = long_term_emb[sorted_idx]

        packed = pack_padded_sequence(
            sorted_vectors, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        h0 = sorted_long_term.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        _, (last_hidden, _) = self.lstm(packed, (h0, c0))
        u_s_sorted = last_hidden.squeeze(0)

        _, unsorted_idx = torch.sort(sorted_idx)
        return self.linear(u_s_sorted[unsorted_idx])

# ============================================================
# DNN CLICK PREDICTOR
# ============================================================
class DNNClickPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, candidate_news_vector, user_vector):
        return self.dnn(torch.cat((candidate_news_vector, user_vector), dim=1)).squeeze(1)

# ============================================================
# Co_NAML_LSTUR MODEL
# ============================================================
class Co_NAML_LSTUR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.user_embedding = nn.Embedding(
            config['num_users'], int(config['num_filters'] * 1.5), padding_idx=0
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self.attention = DKNAttention(config)
        self.click_predictor = DNNClickPredictor(input_size=config['num_filters'] * 2)

    def forward(self, user, clicked_news_length, candidate_news_vectors, clicked_news_vectors):
        B, N, D = candidate_news_vectors.shape
        S = clicked_news_vectors.shape[1]

        u_l = F.dropout(self.user_embedding(user), p=self.config['masking_probability'], training=self.training)
        u_s = self.user_encoder(u_l, clicked_news_length, clicked_news_vectors)

        candidate_flat = candidate_news_vectors.reshape(B * N, D)
        clicked_expanded = clicked_news_vectors.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, S, D)
        u_att = self.attention(candidate_flat, clicked_expanded).view(B, N, D)

        user_vector = u_s.unsqueeze(1) * u_att
        scores = self.click_predictor(
            candidate_news_vectors.reshape(B * N, D),
            user_vector.reshape(B * N, D)
        ).view(B, N)
        return scores

# ============================================================
# DATASET CLASS
# ============================================================
class ImpressionDataset(Dataset):
    def __init__(self, behavior_path, news_dict, config, mode='train'):
        self.data = pd.read_csv(
            behavior_path, sep="\t", header=None,
            names=["impression_id","user_id","time","history","impressions"]
        )
        self.news_dict = news_dict
        self.valid_news = list(news_dict.keys())
        self.config = config
        self.mode = mode
        self.max_history = config['max_history']
        self.K = config['negative_sampling_ratio']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = user2idx[row["user_id"]]

        history = [] if pd.isna(row["history"]) else row["history"].split()
        history = history[-self.max_history:]
        clicked = []
        for nid in history:
            if nid in self.news_dict:
                clicked.append(self.news_dict[nid])
            else:
                clicked.append(self.news_dict[random.choice(self.valid_news)])
        clicked_len = max(len(clicked), 1)

        impressions = row["impressions"].split()
        if self.mode == 'train':
            pos, neg = [], []
            for imp in impressions:
                nid, lbl = imp.split("-")
                if nid in self.news_dict:
                    if int(lbl) == 1:
                        pos.append(self.news_dict[nid])
                    else:
                        neg.append(self.news_dict[nid])
            if pos and neg:
                pos_sample = random.choice(pos)
                neg_samples = random.sample(neg, min(self.K, len(neg)))
                while len(neg_samples) < self.K:
                    neg_samples.append(random.choice(neg))
                candidates = [pos_sample] + neg_samples
                labels = [1] + [0]*self.K
            else:
                candidates = [self.news_dict[random.choice(self.valid_news)] for _ in range(1+self.K)]
                labels = [0]*(1+self.K)
        else:
            candidates, labels = [], []
            for imp in impressions:
                nid, lbl = imp.split("-")
                if nid in self.news_dict:
                    candidates.append(self.news_dict[nid])
                    labels.append(int(lbl))

        return (
            torch.tensor(user),
            torch.stack(clicked) if clicked else torch.zeros(1, config['num_filters']),
            torch.stack(candidates),
            torch.tensor(clicked_len),
            torch.tensor(labels, dtype=torch.float32),
            len(labels)
        )

# ============================================================
# COLLATE FUNCTION
# ============================================================
def collate_fn(batch):
    users, histories, candidates, hist_lens, labels, lens = zip(*batch)
    users = torch.stack(users)
    hist_lens = torch.stack(hist_lens)
    lens = torch.tensor(lens)

    B, D = len(histories), histories[0].size(1)
    max_hist = max(h.size(0) for h in histories)
    padded_hist = torch.zeros(B, max_hist, D)
    for i, h in enumerate(histories):
        padded_hist[i, :h.size(0)] = h

    max_cand = max(c.size(0) for c in candidates)
    padded_cand = torch.zeros(B, max_cand, D)
    padded_labels = torch.zeros(B, max_cand)
    for i, (c, l) in enumerate(zip(candidates, labels)):
        padded_cand[i, :c.size(0)] = c
        padded_labels[i, :l.size(0)] = l

    return users, padded_hist, padded_cand, hist_lens, padded_labels, lens

# ============================================================
# DATALOADERS
# ============================================================
train_set = ImpressionDataset(TRAIN_BEHAVIOR_PATH, news_vectors, config, mode='train')
val_set   = ImpressionDataset(VAL_BEHAVIOR_PATH,   news_vectors, config, mode='val')
test_set  = ImpressionDataset(VAL_BEHAVIOR_PATH,   news_vectors, config, mode='test')

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
val_loader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)
test_loader  = DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)

# ============================================================
# METRICS
# ============================================================
def mrr_score(y_true, y_pred):
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    for i, v in enumerate(y_true_sorted):
        if v == 1:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_score(y_true, y_pred, k):
    if len(y_true) < k:
        k = len(y_true)
    order = np.argsort(-y_pred)[:k]
    dcg = np.sum((2**y_true[order] - 1) / np.log2(np.arange(2, k + 2)))
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum((2**y_true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0

# ============================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    for users, history, candidates, hist_len, labels, lens in tqdm(loader, desc="Training"):
        users, history, candidates = users.to(device), history.to(device), candidates.to(device)
        hist_len, labels, lens = hist_len.to(device), labels.to(device), lens.to(device)

        optimizer.zero_grad()
        predictions = model(users, hist_len, candidates, history)
        mask = torch.arange(labels.size(1), device=device).unsqueeze(0) < lens.unsqueeze(1)
        loss = criterion(predictions[mask], labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    if scheduler:
        scheduler.step()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    with torch.no_grad():
        for users, history, candidates, hist_len, labels, lens in tqdm(loader, desc="Evaluating"):
            users, history, candidates = users.to(device), history.to(device), candidates.to(device)
            hist_len, labels, lens = hist_len.to(device), labels.to(device), lens.to(device)

            predictions = model(users, hist_len, candidates, history)
            preds = torch.sigmoid(predictions).cpu().numpy()
            labels_np = labels.cpu().numpy()
            lens_np = lens.cpu().numpy()

            for y_pred, y_true, L in zip(preds, labels_np, lens_np):
                if L == 0:
                    continue
                y_pred_valid = y_pred[:L]
                y_true_valid = y_true[:L]

                mrrs.append(mrr_score(y_true_valid, y_pred_valid))
                ndcg5s.append(ndcg_score(y_true_valid, y_pred_valid, k=5))
                ndcg10s.append(ndcg_score(y_true_valid, y_pred_valid, k=10))

                if len(np.unique(y_true_valid)) >= 2:
                    try:
                        aucs.append(roc_auc_score(y_true_valid, y_pred_valid))
                    except ValueError:
                        pass

    return (
        np.mean(aucs) if aucs else 0.0,
        np.mean(mrrs) if mrrs else 0.0,
        np.mean(ndcg5s) if ndcg5s else 0.0,
        np.mean(ndcg10s) if ndcg10s else 0.0
    )

# ============================================================
# MODEL INITIALIZATION
# ============================================================
model = Co_NAML_LSTUR(config).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TRAINING LOOP
# ============================================================
best_mrr = 0
for epoch in range(1, config['epochs']+1):
    print(f"\nEpoch {epoch}/{config['epochs']}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
    auc, mrr, ndcg5, ndcg10 = evaluate(model, val_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUC: {auc:.4f}, MRR: {mrr:.4f}, nDCG@5: {ndcg5:.4f}, nDCG@10: {ndcg10:.4f}")

    if mrr > best_mrr:
        best_mrr = mrr
        torch.save(model.state_dict(), 'best_lstur_model.pt')
        print(f"Best MRR: {best_mrr:.4f} - Model saved!")

# ============================================================
# TEST EVALUATION
# ============================================================
print("\nTest Evaluation:")
model.load_state_dict(torch.load('best_lstur_model.pt'))
test_auc, test_mrr, test_ndcg5, test_ndcg10 = evaluate(model, test_loader)
print(f"Test AUC: {test_auc:.4f}, MRR: {test_mrr:.4f}, nDCG@5: {test_ndcg5:.4f}, nDCG@10: {test_ndcg10:.4f}")