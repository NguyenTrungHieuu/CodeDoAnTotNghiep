# ========================================================
# NAML + DistilBERT - Đúng chuẩn paper Co-NAML-LSTUR
# Train = 30% | Val = 25% | Test = 26% của bộ val gốc
# ========================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import warnings
from pathlib import Path
import gc

warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
gc.collect()

# ====================== ĐƯỜNG DẪN ======================
BASE_PATH = Path("/kaggle/input/datasets/neyugn12")
NEWS_TRAIN_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_train" / "news.tsv"
BEHAVIORS_TRAIN_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_train" / "behaviors.tsv"
NEWS_DEV_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_val" / "news.tsv"
BEHAVIORS_DEV_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_val" / "behaviors.tsv"

# ========================= CACHE DISTILBERT =========================
CACHE_DIR = "/kaggle/working/distilbert_cache"
from transformers import DistilBertTokenizer, DistilBertModel

if os.path.exists(CACHE_DIR) and os.path.exists(os.path.join(CACHE_DIR, "pytorch_model.bin")):
    print("Loading DistilBERT from CACHE...")
    tokenizer = DistilBertTokenizer.from_pretrained(CACHE_DIR)
    distilbert = DistilBertModel.from_pretrained(CACHE_DIR).to(device)
else:
    print("Downloading & caching DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer.save_pretrained(CACHE_DIR)
    distilbert.save_pretrained(CACHE_DIR)
    distilbert = distilbert.to(device)

for param in distilbert.parameters():
    param.requires_grad = False
distilbert.eval()

HIDDEN_SIZE = distilbert.config.hidden_size

# ========================= HYPERPARAMETERS =========================
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 20
NEGATIVE_RATIO = 4
HISTORY_LEN = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
EPOCHS = 5
NUM_FILTERS = 256
QUERY_VECTOR_DIM = 128
MAX_TITLE_LEN = 30
MAX_ABS_LEN = 50

# ========================= CLASSES =========================
class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super().__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.query_vector = nn.Parameter(torch.randn(query_vector_dim, 1))

    def forward(self, candidate_vector):
        projected = torch.tanh(self.linear(candidate_vector))
        attention_scores = torch.matmul(projected, self.query_vector).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_sum = torch.bmm(candidate_vector.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        return weighted_sum

class TextEncoder(nn.Module):
    def __init__(self, distilbert, num_filters, query_vector_dim, dropout_probability):
        super().__init__()
        self.distilbert = distilbert
        self.dropout_probability = dropout_probability
        self.projection = nn.Linear(HIDDEN_SIZE, num_filters)
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = bert_output.last_hidden_state
        projected = self.projection(token_emb)
        projected = F.dropout(projected, p=self.dropout_probability, training=self.training)
        return self.additive_attention(projected)

class ElementEncoder(nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super().__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        element = torch.clamp(element, 0, self.embedding.num_embeddings - 1)
        return F.relu(self.linear(self.embedding(element)))

class NewsEncoder(nn.Module):
    def __init__(self, distilbert, num_categories, num_subcategories):
        super().__init__()
        self.text_encoders = nn.ModuleDict({
            "title": TextEncoder(distilbert, NUM_FILTERS, QUERY_VECTOR_DIM, DROPOUT_RATE),
            "abstract": TextEncoder(distilbert, NUM_FILTERS, QUERY_VECTOR_DIM, DROPOUT_RATE)
        })
        self.cat_embedding = nn.Embedding(num_categories, NUM_FILTERS, padding_idx=0)
        self.subcat_embedding = nn.Embedding(num_subcategories, NUM_FILTERS, padding_idx=0)
        self.element_encoders = nn.ModuleDict({
            "category": ElementEncoder(self.cat_embedding, NUM_FILTERS, NUM_FILTERS),
            "subcategory": ElementEncoder(self.subcat_embedding, NUM_FILTERS, NUM_FILTERS)
        })
        self.final_attention = AdditiveAttention(QUERY_VECTOR_DIM, NUM_FILTERS)

    def forward(self, news):
        B = news["title"].size(0)
        num_news = news["title"].size(1)
        title_flat = news["title"].view(B * num_news, -1)
        title_mask_flat = news["title_mask"].view(B * num_news, -1)
        abs_flat = news["abstract"].view(B * num_news, -1)
        abs_mask_flat = news["abstract_mask"].view(B * num_news, -1)
        cat_flat = news["category"].view(B * num_news)
        subcat_flat = news["subcategory"].view(B * num_news)

        title_vec = self.text_encoders["title"](title_flat, title_mask_flat)
        abs_vec = self.text_encoders["abstract"](abs_flat, abs_mask_flat)
        cat_vec = self.element_encoders["category"](cat_flat)
        subcat_vec = self.element_encoders["subcategory"](subcat_flat)

        all_vectors = torch.stack([title_vec, abs_vec, cat_vec, subcat_vec], dim=1)
        final_flat = self.final_attention(all_vectors)
        return final_flat.view(B, num_news, -1)

class UserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.additive_attention = AdditiveAttention(QUERY_VECTOR_DIM, NUM_FILTERS)

    def forward(self, clicked_news_vector):
        return self.additive_attention(clicked_news_vector)

class NAML(nn.Module):
    def __init__(self, num_categories, num_subcategories, distilbert):
        super().__init__()
        self.news_encoder = NewsEncoder(distilbert, num_categories, num_subcategories)
        self.user_encoder = UserEncoder()

    def forward(self, h_title, h_title_mask, h_abs, h_abs_mask, h_cat, h_sub,
                c_title, c_title_mask, c_abs, c_abs_mask, c_cat, c_sub):
        h_news = {"title": h_title, "title_mask": h_title_mask,
                  "abstract": h_abs, "abstract_mask": h_abs_mask,
                  "category": h_cat, "subcategory": h_sub}
        h_vecs = self.news_encoder(h_news)
        user_vec = self.user_encoder(h_vecs)

        c_news = {"title": c_title, "title_mask": c_title_mask,
                  "abstract": c_abs, "abstract_mask": c_abs_mask,
                  "category": c_cat, "subcategory": c_sub}
        c_vecs = self.news_encoder(c_news)
        scores = torch.bmm(c_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores

# ========================= LOAD DATA =========================
print("Loading MIND-small dataset...")
news_columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
behaviors_columns = ["impression_id", "user_id", "time", "history", "impressions"]

news_train = pd.read_csv(NEWS_TRAIN_PATH, sep="\t", names=news_columns)
news_dev = pd.read_csv(NEWS_DEV_PATH, sep="\t", names=news_columns)
behaviors_train = pd.read_csv(BEHAVIORS_TRAIN_PATH, sep="\t", names=behaviors_columns)
behaviors_dev = pd.read_csv(BEHAVIORS_DEV_PATH, sep="\t", names=behaviors_columns)

news_df = pd.concat([news_train, news_dev], ignore_index=True).drop_duplicates(subset=['news_id']).reset_index(drop=True)
news_df["abstract"] = news_df["abstract"].fillna("")
news_df["title"] = news_df["title"].fillna("")
news_df["category"] = news_df["category"].fillna("unknown")
news_df["subcategory"] = news_df["subcategory"].fillna("unknown")

print("Tokenizing news with DistilBERT (batch mode)...")
title_enc = tokenizer(news_df["title"].tolist(), max_length=MAX_TITLE_LEN, padding='max_length', truncation=True, return_tensors='pt')
abs_enc = tokenizer(news_df["abstract"].tolist(), max_length=MAX_ABS_LEN, padding='max_length', truncation=True, return_tensors='pt')

news_df["title_ids"] = [ids.tolist() for ids in title_enc['input_ids']]
news_df["title_mask"] = [mask.tolist() for mask in title_enc['attention_mask']]
news_df["abstract_ids"] = [ids.tolist() for ids in abs_enc['input_ids']]
news_df["abstract_mask"] = [mask.tolist() for mask in abs_enc['attention_mask']]

cat_dict = {c: i+1 for i, c in enumerate(news_df["category"].unique())}
cat_dict["unknown"] = 0
subcat_dict = {c: i+1 for i, c in enumerate(news_df["subcategory"].unique())}
subcat_dict["unknown"] = 0

news_df["cat_id"] = news_df["category"].map(cat_dict).astype(int)
news_df["subcat_id"] = news_df["subcategory"].map(subcat_dict).astype(int)
news_index = {nid: idx for idx, nid in enumerate(news_df["news_id"])}

def parse_behaviors(df):
    df["history_parsed"] = df["history"].fillna("").str.split()
    df["impressions_parsed"] = df["impressions"].str.split().apply(
        lambda x: [(item.split("-")[0], int(item.split("-")[1])) for item in x if "-" in item])
    return df

behaviors_train = parse_behaviors(behaviors_train)
behaviors_dev = parse_behaviors(behaviors_dev)

# ====================== LẤY 30% TRAIN ======================
print("Lấy 30% của full train dataset...")
behaviors_train = behaviors_train.sample(frac=0.3, random_state=42).reset_index(drop=True)
print(f"Train sau khi lấy 30%: {len(behaviors_train):,} impressions")

# ====================== CHIA VAL & TEST ======================
print("Chia Val gốc thành Val + Test...")
behaviors_dev = behaviors_dev.sample(frac=1, random_state=42).reset_index(drop=True)

val_size = int(len(behaviors_dev) * 0.25)   # 25% của val gốc
test_size = int(len(behaviors_dev) * 0.26)  # 26% của val gốc

val_df = behaviors_dev.iloc[:val_size].reset_index(drop=True)
test_df = behaviors_dev.iloc[val_size:val_size + test_size].reset_index(drop=True)

print(f"Val = {len(val_df):,} | Test = {len(test_df):,}")

train_df = behaviors_train.reset_index(drop=True)
print(f"TRAINING 30% DATASET → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Total news: {len(news_df):,}")

# ========================= MODEL =========================
model = NAML(len(cat_dict), len(subcat_dict), distilbert).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
print(f"NAML + DistilBERT (theo paper) | Params: {sum(p.numel() for p in model.parameters()):,}")

def naml_loss(scores, labels):
    log_probs = F.log_softmax(scores, dim=1)
    return -log_probs[:, 0].mean()

# ========================= PREPARE BATCH =========================
def prepare_train_batch(batch_df, news_df, news_index):
    h_title_list, h_title_mask_list = [], []
    h_abs_list, h_abs_mask_list = [], []
    h_cat_list, h_sub_list = [], []
    c_title_list, c_title_mask_list = [], []
    c_abs_list, c_abs_mask_list = [], []
    c_cat_list, c_sub_list = [], []
    label_list = []
    for _, row in batch_df.iterrows():
        hist = row["history_parsed"][-HISTORY_LEN:]
        imps = row["impressions_parsed"]
        if len(hist) == 0 or len(imps) < 2: continue
        h_t, h_tm, h_a, h_am, h_c, h_s = [], [], [], [], [], []
        for nid in hist:
            if nid in news_index:
                idx = news_index[nid]
                h_t.append(news_df.iloc[idx]["title_ids"])
                h_tm.append(news_df.iloc[idx]["title_mask"])
                h_a.append(news_df.iloc[idx]["abstract_ids"])
                h_am.append(news_df.iloc[idx]["abstract_mask"])
                h_c.append(news_df.iloc[idx]["cat_id"])
                h_s.append(news_df.iloc[idx]["subcat_id"])
        while len(h_t) < HISTORY_LEN:
            h_t.append([0]*MAX_TITLE_LEN)
            h_tm.append([0]*MAX_TITLE_LEN)
            h_a.append([0]*MAX_ABS_LEN)
            h_am.append([0]*MAX_ABS_LEN)
            h_c.append(0)
            h_s.append(0)
        positives = [nid for nid, lbl in imps if lbl == 1]
        if not positives: continue
        pos_nid = random.choice(positives)
        negs = [nid for nid, lbl in imps if lbl == 0 and nid not in hist and nid != pos_nid]
        selected_negs = random.sample(negs, NEGATIVE_RATIO) if len(negs) >= NEGATIVE_RATIO else \
                        negs + random.choices([nid for nid in news_index if nid not in hist and nid != pos_nid],
                                              k=NEGATIVE_RATIO - len(negs))
        cands = [pos_nid] + selected_negs[:NEGATIVE_RATIO]
        c_t, c_tm, c_a, c_am, c_c, c_s = [], [], [], [], [], []
        for nid in cands:
            idx = news_index[nid]
            c_t.append(news_df.iloc[idx]["title_ids"])
            c_tm.append(news_df.iloc[idx]["title_mask"])
            c_a.append(news_df.iloc[idx]["abstract_ids"])
            c_am.append(news_df.iloc[idx]["abstract_mask"])
            c_c.append(news_df.iloc[idx]["cat_id"])
            c_s.append(news_df.iloc[idx]["subcat_id"])
        labels = [1] + [0] * NEGATIVE_RATIO
        h_title_list.append(h_t)
        h_title_mask_list.append(h_tm)
        h_abs_list.append(h_a)
        h_abs_mask_list.append(h_am)
        h_cat_list.append(h_c)
        h_sub_list.append(h_s)
        c_title_list.append(c_t)
        c_title_mask_list.append(c_tm)
        c_abs_list.append(c_a)
        c_abs_mask_list.append(c_am)
        c_cat_list.append(c_c)
        c_sub_list.append(c_s)
        label_list.append(labels)
    if not h_title_list: return None
    return {
        "h_title": torch.LongTensor(h_title_list).to(device),
        "h_title_mask": torch.LongTensor(h_title_mask_list).to(device),
        "h_abs": torch.LongTensor(h_abs_list).to(device),
        "h_abs_mask": torch.LongTensor(h_abs_mask_list).to(device),
        "h_cat": torch.LongTensor(h_cat_list).to(device),
        "h_sub": torch.LongTensor(h_sub_list).to(device),
        "c_title": torch.LongTensor(c_title_list).to(device),
        "c_title_mask": torch.LongTensor(c_title_mask_list).to(device),
        "c_abs": torch.LongTensor(c_abs_list).to(device),
        "c_abs_mask": torch.LongTensor(c_abs_mask_list).to(device),
        "c_cat": torch.LongTensor(c_cat_list).to(device),
        "c_sub": torch.LongTensor(c_sub_list).to(device),
        "labels": torch.LongTensor(label_list).to(device)
    }

def prepare_eval_batch(batch_df, news_df, news_index):
    h_title_list, h_title_mask_list = [], []
    h_abs_list, h_abs_mask_list = [], []
    h_cat_list, h_sub_list = [], []
    c_title_list, c_title_mask_list = [], []
    c_abs_list, c_abs_mask_list = [], []
    c_cat_list, c_sub_list = [], []
    label_list = []
    for _, row in batch_df.iterrows():
        hist = row["history_parsed"][-HISTORY_LEN:]
        imps = row["impressions_parsed"]
        if len(hist) == 0 or len(imps) == 0: continue
        h_t, h_tm, h_a, h_am, h_c, h_s = [], [], [], [], [], []
        for nid in hist:
            if nid in news_index:
                idx = news_index[nid]
                h_t.append(news_df.iloc[idx]["title_ids"])
                h_tm.append(news_df.iloc[idx]["title_mask"])
                h_a.append(news_df.iloc[idx]["abstract_ids"])
                h_am.append(news_df.iloc[idx]["abstract_mask"])
                h_c.append(news_df.iloc[idx]["cat_id"])
                h_s.append(news_df.iloc[idx]["subcat_id"])
        while len(h_t) < HISTORY_LEN:
            h_t.append([0]*MAX_TITLE_LEN)
            h_tm.append([0]*MAX_TITLE_LEN)
            h_a.append([0]*MAX_ABS_LEN)
            h_am.append([0]*MAX_ABS_LEN)
            h_c.append(0)
            h_s.append(0)
        c_t, c_tm, c_a, c_am, c_c, c_s = [], [], [], [], [], []
        labels = []
        for nid, lbl in imps:
            if nid in news_index:
                idx = news_index[nid]
                c_t.append(news_df.iloc[idx]["title_ids"])
                c_tm.append(news_df.iloc[idx]["title_mask"])
                c_a.append(news_df.iloc[idx]["abstract_ids"])
                c_am.append(news_df.iloc[idx]["abstract_mask"])
                c_c.append(news_df.iloc[idx]["cat_id"])
                c_s.append(news_df.iloc[idx]["subcat_id"])
                labels.append(lbl)
        if len(c_t) == 0: continue
        h_title_list.append(h_t)
        h_title_mask_list.append(h_tm)
        h_abs_list.append(h_a)
        h_abs_mask_list.append(h_am)
        h_cat_list.append(h_c)
        h_sub_list.append(h_s)
        c_title_list.append(c_t)
        c_title_mask_list.append(c_tm)
        c_abs_list.append(c_a)
        c_abs_mask_list.append(c_am)
        c_cat_list.append(c_c)
        c_sub_list.append(c_s)
        label_list.append(labels)
    if not h_title_list: return None
    max_cands = max(len(lst) for lst in c_title_list)
    for i in range(len(c_title_list)):
        pad_len = max_cands - len(c_title_list[i])
        if pad_len > 0:
            c_title_list[i] += [[0] * MAX_TITLE_LEN] * pad_len
            c_title_mask_list[i] += [[0] * MAX_TITLE_LEN] * pad_len
            c_abs_list[i] += [[0] * MAX_ABS_LEN] * pad_len
            c_abs_mask_list[i] += [[0] * MAX_ABS_LEN] * pad_len
            c_cat_list[i] += [0] * pad_len
            c_sub_list[i] += [0] * pad_len
    return {
        "h_title": torch.LongTensor(h_title_list).to(device),
        "h_title_mask": torch.LongTensor(h_title_mask_list).to(device),
        "h_abs": torch.LongTensor(h_abs_list).to(device),
        "h_abs_mask": torch.LongTensor(h_abs_mask_list).to(device),
        "h_cat": torch.LongTensor(h_cat_list).to(device),
        "h_sub": torch.LongTensor(h_sub_list).to(device),
        "c_title": torch.LongTensor(c_title_list).to(device),
        "c_title_mask": torch.LongTensor(c_title_mask_list).to(device),
        "c_abs": torch.LongTensor(c_abs_list).to(device),
        "c_abs_mask": torch.LongTensor(c_abs_mask_list).to(device),
        "c_cat": torch.LongTensor(c_cat_list).to(device),
        "c_sub": torch.LongTensor(c_sub_list).to(device),
        "labels": label_list
    }

# ========================= EVALUATE =========================
@torch.no_grad()
def evaluate(df_split):
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    print(f"Evaluating {len(df_split):,} impressions with EVAL_BATCH_SIZE = {EVAL_BATCH_SIZE}...")

    for i in tqdm(range(0, len(df_split), EVAL_BATCH_SIZE), desc="Evaluating"):
        torch.cuda.empty_cache()
        gc.collect()
        
        batch_df = df_split.iloc[i:i + EVAL_BATCH_SIZE]
        batch = prepare_eval_batch(batch_df, news_df, news_index)
        if batch is None:
            continue

        scores = model(batch["h_title"], batch["h_title_mask"], batch["h_abs"], batch["h_abs_mask"],
                       batch["h_cat"], batch["h_sub"],
                       batch["c_title"], batch["c_title_mask"], batch["c_abs"], batch["c_abs_mask"],
                       batch["c_cat"], batch["c_sub"])

        torch.cuda.empty_cache()
        gc.collect()

        for j in range(len(batch["labels"])):
            true_labels = np.array(batch["labels"][j])
            num_real_cands = len(true_labels)
            score_vals = scores[j, :num_real_cands].cpu().numpy()

            if len(set(true_labels)) < 2:
                continue

            aucs.append(roc_auc_score(true_labels, score_vals))

            ranks = np.argsort(-score_vals)
            first_pos = np.where(true_labels[ranks] == 1)[0]
            if len(first_pos) > 0:
                mrrs.append(1.0 / (first_pos[0] + 1))

            def dcg(r, k):
                r = np.asarray(r)[:k]
                return np.sum(r / np.log2(np.arange(2, len(r) + 2)))

            def ndcg(r, k):
                dcg_val = dcg(r, k)
                idcg = dcg(sorted(r, reverse=True), k)
                return dcg_val / idcg if idcg > 0 else 0.0

            rel = true_labels[ranks]
            ndcg5s.append(ndcg(rel, 5))
            ndcg10s.append(ndcg(rel, 10))

    return {
        "AUC": np.mean(aucs) if aucs else 0,
        "MRR": np.mean(mrrs) if mrrs else 0,
        "nDCG@5": np.mean(ndcg5s) if ndcg5s else 0,
        "nDCG@10": np.mean(ndcg10s) if ndcg10s else 0
    }

# ========================= TRAINING =========================
def train_epoch():
    model.train()
    total_loss = 0.0
    torch.cuda.empty_cache()
    gc.collect()
    for i in tqdm(range(0, len(train_df), BATCH_SIZE), desc="Training"):
        batch_df = train_df.iloc[i:i + BATCH_SIZE]
        batch = prepare_train_batch(batch_df, news_df, news_index)
        if batch is None: continue
        with torch.cuda.amp.autocast():
            scores = model(batch["h_title"], batch["h_title_mask"], batch["h_abs"], batch["h_abs_mask"],
                           batch["h_cat"], batch["h_sub"],
                           batch["c_title"], batch["c_title_mask"], batch["c_abs"], batch["c_abs_mask"],
                           batch["c_cat"], batch["c_sub"])
            loss = naml_loss(scores, batch["labels"])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / (len(train_df) // BATCH_SIZE + 1)

# ========================= TRAINING LOOP =========================
best_auc = 0.0
print("\n" + "="*80)
print("TRAINING 30% DATASET NAML + DistilBERT")
print("="*80)

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== EPOCH {epoch}/{EPOCHS} ===")
    train_loss = train_epoch()
    print(f"Train Loss: {train_loss:.4f}")

    val_metrics = evaluate(val_df)
    print(f"Val → AUC: {val_metrics['AUC']:.4f} | MRR: {val_metrics['MRR']:.4f} | "
          f"nDCG@5: {val_metrics['nDCG@5']:.4f} | nDCG@10: {val_metrics['nDCG@10']:.4f}")

    if val_metrics["AUC"] > best_auc:
        best_auc = val_metrics["AUC"]
        torch.save(model.state_dict(), "naml_distilbert_best.pt")
        print("Saved best model")

print("\n=== FINAL TEST EVALUATION ===")
test_metrics = evaluate(test_df)
print(f"Test → AUC: {test_metrics['AUC']:.4f} | MRR: {test_metrics['MRR']:.4f} | "
      f"nDCG@5: {test_metrics['nDCG@5']:.4f} | nDCG@10: {test_metrics['nDCG@10']:.4f}")

print(f"\nBest Validation AUC: {best_auc:.4f}")
print("HOÀN TẤT!")

# ========================================================
# NAML + DistilBERT - Đúng chuẩn paper Co-NAML-LSTUR
# Train = 30% | Val = 25% | Test = 26% của bộ val gốc
# TIẾP TỤC TRAIN TỪ EPOCH 5 (RESUME)
# ========================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import warnings
from pathlib import Path
import gc
warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
gc.collect()
# ====================== ĐƯỜNG DẪN ======================
BASE_PATH = Path("/kaggle/input/datasets/neyugn12")
NEWS_TRAIN_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_train" / "news.tsv"
BEHAVIORS_TRAIN_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_train" / "behaviors.tsv"
NEWS_DEV_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_val" / "news.tsv"
BEHAVIORS_DEV_PATH = BASE_PATH / "dataset-mind-small" / "MINDsmall_val" / "behaviors.tsv"
# ========================= CACHE DISTILBERT =========================
CACHE_DIR = "/kaggle/working/distilbert_cache"
from transformers import DistilBertTokenizer, DistilBertModel
if os.path.exists(CACHE_DIR) and os.path.exists(os.path.join(CACHE_DIR, "pytorch_model.bin")):
    print("Loading DistilBERT from CACHE...")
    tokenizer = DistilBertTokenizer.from_pretrained(CACHE_DIR)
    distilbert = DistilBertModel.from_pretrained(CACHE_DIR).to(device)
else:
    print("Downloading & caching DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer.save_pretrained(CACHE_DIR)
    distilbert.save_pretrained(CACHE_DIR)
    distilbert = distilbert.to(device)
for param in distilbert.parameters():
    param.requires_grad = False
distilbert.eval()
HIDDEN_SIZE = distilbert.config.hidden_size
# ========================= HYPERPARAMETERS =========================
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 20
NEGATIVE_RATIO = 4
HISTORY_LEN = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
EPOCHS = 5
NUM_FILTERS = 256
QUERY_VECTOR_DIM = 128
MAX_TITLE_LEN = 30
MAX_ABS_LEN = 50
# ========================= CLASSES =========================
class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super().__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.query_vector = nn.Parameter(torch.randn(query_vector_dim, 1))
    def forward(self, candidate_vector):
        projected = torch.tanh(self.linear(candidate_vector))
        attention_scores = torch.matmul(projected, self.query_vector).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_sum = torch.bmm(candidate_vector.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        return weighted_sum
class TextEncoder(nn.Module):
    def __init__(self, distilbert, num_filters, query_vector_dim, dropout_probability):
        super().__init__()
        self.distilbert = distilbert
        self.dropout_probability = dropout_probability
        self.projection = nn.Linear(HIDDEN_SIZE, num_filters)
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = bert_output.last_hidden_state
        projected = self.projection(token_emb)
        projected = F.dropout(projected, p=self.dropout_probability, training=self.training)
        return self.additive_attention(projected)
class ElementEncoder(nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super().__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)
    def forward(self, element):
        element = torch.clamp(element, 0, self.embedding.num_embeddings - 1)
        return F.relu(self.linear(self.embedding(element)))
class NewsEncoder(nn.Module):
    def __init__(self, distilbert, num_categories, num_subcategories):
        super().__init__()
        self.text_encoders = nn.ModuleDict({
            "title": TextEncoder(distilbert, NUM_FILTERS, QUERY_VECTOR_DIM, DROPOUT_RATE),
            "abstract": TextEncoder(distilbert, NUM_FILTERS, QUERY_VECTOR_DIM, DROPOUT_RATE)
        })
        self.cat_embedding = nn.Embedding(num_categories, NUM_FILTERS, padding_idx=0)
        self.subcat_embedding = nn.Embedding(num_subcategories, NUM_FILTERS, padding_idx=0)
        self.element_encoders = nn.ModuleDict({
            "category": ElementEncoder(self.cat_embedding, NUM_FILTERS, NUM_FILTERS),
            "subcategory": ElementEncoder(self.subcat_embedding, NUM_FILTERS, NUM_FILTERS)
        })
        self.final_attention = AdditiveAttention(QUERY_VECTOR_DIM, NUM_FILTERS)
    def forward(self, news):
        B = news["title"].size(0)
        num_news = news["title"].size(1)
        title_flat = news["title"].view(B * num_news, -1)
        title_mask_flat = news["title_mask"].view(B * num_news, -1)
        abs_flat = news["abstract"].view(B * num_news, -1)
        abs_mask_flat = news["abstract_mask"].view(B * num_news, -1)
        cat_flat = news["category"].view(B * num_news)
        subcat_flat = news["subcategory"].view(B * num_news)
        title_vec = self.text_encoders["title"](title_flat, title_mask_flat)
        abs_vec = self.text_encoders["abstract"](abs_flat, abs_mask_flat)
        cat_vec = self.element_encoders["category"](cat_flat)
        subcat_vec = self.element_encoders["subcategory"](subcat_flat)
        all_vectors = torch.stack([title_vec, abs_vec, cat_vec, subcat_vec], dim=1)
        final_flat = self.final_attention(all_vectors)
        return final_flat.view(B, num_news, -1)
class UserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.additive_attention = AdditiveAttention(QUERY_VECTOR_DIM, NUM_FILTERS)
    def forward(self, clicked_news_vector):
        return self.additive_attention(clicked_news_vector)
class NAML(nn.Module):
    def __init__(self, num_categories, num_subcategories, distilbert):
        super().__init__()
        self.news_encoder = NewsEncoder(distilbert, num_categories, num_subcategories)
        self.user_encoder = UserEncoder()
    def forward(self, h_title, h_title_mask, h_abs, h_abs_mask, h_cat, h_sub,
                c_title, c_title_mask, c_abs, c_abs_mask, c_cat, c_sub):
        h_news = {"title": h_title, "title_mask": h_title_mask,
                  "abstract": h_abs, "abstract_mask": h_abs_mask,
                  "category": h_cat, "subcategory": h_sub}
        h_vecs = self.news_encoder(h_news)
        user_vec = self.user_encoder(h_vecs)
        c_news = {"title": c_title, "title_mask": c_title_mask,
                  "abstract": c_abs, "abstract_mask": c_abs_mask,
                  "category": c_cat, "subcategory": c_sub}
        c_vecs = self.news_encoder(c_news)
        scores = torch.bmm(c_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores
# ========================= LOAD DATA =========================
print("Loading MIND-small dataset...")
news_columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
behaviors_columns = ["impression_id", "user_id", "time", "history", "impressions"]
news_train = pd.read_csv(NEWS_TRAIN_PATH, sep="\t", names=news_columns)
news_dev = pd.read_csv(NEWS_DEV_PATH, sep="\t", names=news_columns)
behaviors_train = pd.read_csv(BEHAVIORS_TRAIN_PATH, sep="\t", names=behaviors_columns)
behaviors_dev = pd.read_csv(BEHAVIORS_DEV_PATH, sep="\t", names=behaviors_columns)
news_df = pd.concat([news_train, news_dev], ignore_index=True).drop_duplicates(subset=['news_id']).reset_index(drop=True)
news_df["abstract"] = news_df["abstract"].fillna("")
news_df["title"] = news_df["title"].fillna("")
news_df["category"] = news_df["category"].fillna("unknown")
news_df["subcategory"] = news_df["subcategory"].fillna("unknown")
print("Tokenizing news with DistilBERT (batch mode)...")
title_enc = tokenizer(news_df["title"].tolist(), max_length=MAX_TITLE_LEN, padding='max_length', truncation=True, return_tensors='pt')
abs_enc = tokenizer(news_df["abstract"].tolist(), max_length=MAX_ABS_LEN, padding='max_length', truncation=True, return_tensors='pt')
news_df["title_ids"] = [ids.tolist() for ids in title_enc['input_ids']]
news_df["title_mask"] = [mask.tolist() for mask in title_enc['attention_mask']]
news_df["abstract_ids"] = [ids.tolist() for ids in abs_enc['input_ids']]
news_df["abstract_mask"] = [mask.tolist() for mask in abs_enc['attention_mask']]
cat_dict = {c: i+1 for i, c in enumerate(news_df["category"].unique())}
cat_dict["unknown"] = 0
subcat_dict = {c: i+1 for i, c in enumerate(news_df["subcategory"].unique())}
subcat_dict["unknown"] = 0
news_df["cat_id"] = news_df["category"].map(cat_dict).astype(int)
news_df["subcat_id"] = news_df["subcategory"].map(subcat_dict).astype(int)
news_index = {nid: idx for idx, nid in enumerate(news_df["news_id"])}
def parse_behaviors(df):
    df["history_parsed"] = df["history"].fillna("").str.split()
    df["impressions_parsed"] = df["impressions"].str.split().apply(
        lambda x: [(item.split("-")[0], int(item.split("-")[1])) for item in x if "-" in item])
    return df
behaviors_train = parse_behaviors(behaviors_train)
behaviors_dev = parse_behaviors(behaviors_dev)
# ====================== LẤY 30% TRAIN ======================
print("Lấy 30% của full train dataset...")
behaviors_train = behaviors_train.sample(frac=0.3, random_state=42).reset_index(drop=True)
print(f"Train sau khi lấy 30%: {len(behaviors_train):,} impressions")
# ====================== CHIA VAL & TEST ======================
print("Chia Val gốc thành Val + Test...")
behaviors_dev = behaviors_dev.sample(frac=1, random_state=42).reset_index(drop=True)
val_size = int(len(behaviors_dev) * 0.25)
test_size = int(len(behaviors_dev) * 0.26)
val_df = behaviors_dev.iloc[:val_size].reset_index(drop=True)
test_df = behaviors_dev.iloc[val_size:val_size + test_size].reset_index(drop=True)
print(f"Val = {len(val_df):,} | Test = {len(test_df):,}")
train_df = behaviors_train.reset_index(drop=True)
print(f"TRAINING 30% DATASET → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Total news: {len(news_df):,}")
# ========================= MODEL =========================
model = NAML(len(cat_dict), len(subcat_dict), distilbert).to(device)
# ========================= RESUME TỪ EPOCH 5 =========================
LOAD_MODEL_PATH = "/kaggle/input/datasets/neyugn12/save-model-naml/naml_distilbert_best.pt"
if os.path.exists(LOAD_MODEL_PATH):
    print(f"Đang load best model để tiếp tục train từ epoch 5: {LOAD_MODEL_PATH}")
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    print("✓ Đã load model thành công!")
else:
    print("Không tìm thấy file best model, bắt đầu train từ đầu.")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
print(f"NAML + DistilBERT (theo paper) | Params: {sum(p.numel() for p in model.parameters()):,}")
def naml_loss(scores, labels):
    log_probs = F.log_softmax(scores, dim=1)
    return -log_probs[:, 0].mean()
# ========================= PREPARE BATCH =========================
def prepare_train_batch(batch_df, news_df, news_index):
    h_title_list, h_title_mask_list = [], []
    h_abs_list, h_abs_mask_list = [], []
    h_cat_list, h_sub_list = [], []
    c_title_list, c_title_mask_list = [], []
    c_abs_list, c_abs_mask_list = [], []
    c_cat_list, c_sub_list = [], []
    label_list = []
    for _, row in batch_df.iterrows():
        hist = row["history_parsed"][-HISTORY_LEN:]
        imps = row["impressions_parsed"]
        if len(hist) == 0 or len(imps) < 2: continue
        h_t, h_tm, h_a, h_am, h_c, h_s = [], [], [], [], [], []
        for nid in hist:
            if nid in news_index:
                idx = news_index[nid]
                h_t.append(news_df.iloc[idx]["title_ids"])
                h_tm.append(news_df.iloc[idx]["title_mask"])
                h_a.append(news_df.iloc[idx]["abstract_ids"])
                h_am.append(news_df.iloc[idx]["abstract_mask"])
                h_c.append(news_df.iloc[idx]["cat_id"])
                h_s.append(news_df.iloc[idx]["subcat_id"])
        while len(h_t) < HISTORY_LEN:
            h_t.append([0]*MAX_TITLE_LEN)
            h_tm.append([0]*MAX_TITLE_LEN)
            h_a.append([0]*MAX_ABS_LEN)
            h_am.append([0]*MAX_ABS_LEN)
            h_c.append(0)
            h_s.append(0)
        positives = [nid for nid, lbl in imps if lbl == 1]
        if not positives: continue
        pos_nid = random.choice(positives)
        negs = [nid for nid, lbl in imps if lbl == 0 and nid not in hist and nid != pos_nid]
        selected_negs = random.sample(negs, NEGATIVE_RATIO) if len(negs) >= NEGATIVE_RATIO else \
                        negs + random.choices([nid for nid in news_index if nid not in hist and nid != pos_nid],
                                              k=NEGATIVE_RATIO - len(negs))
        cands = [pos_nid] + selected_negs[:NEGATIVE_RATIO]
        c_t, c_tm, c_a, c_am, c_c, c_s = [], [], [], [], [], []
        for nid in cands:
            idx = news_index[nid]
            c_t.append(news_df.iloc[idx]["title_ids"])
            c_tm.append(news_df.iloc[idx]["title_mask"])
            c_a.append(news_df.iloc[idx]["abstract_ids"])
            c_am.append(news_df.iloc[idx]["abstract_mask"])
            c_c.append(news_df.iloc[idx]["cat_id"])
            c_s.append(news_df.iloc[idx]["subcat_id"])
        labels = [1] + [0] * NEGATIVE_RATIO
        h_title_list.append(h_t)
        h_title_mask_list.append(h_tm)
        h_abs_list.append(h_a)
        h_abs_mask_list.append(h_am)
        h_cat_list.append(h_c)
        h_sub_list.append(h_s)
        c_title_list.append(c_t)
        c_title_mask_list.append(c_tm)
        c_abs_list.append(c_a)
        c_abs_mask_list.append(c_am)
        c_cat_list.append(c_c)
        c_sub_list.append(c_s)
        label_list.append(labels)
    if not h_title_list: return None
    return {
        "h_title": torch.LongTensor(h_title_list).to(device),
        "h_title_mask": torch.LongTensor(h_title_mask_list).to(device),
        "h_abs": torch.LongTensor(h_abs_list).to(device),
        "h_abs_mask": torch.LongTensor(h_abs_mask_list).to(device),
        "h_cat": torch.LongTensor(h_cat_list).to(device),
        "h_sub": torch.LongTensor(h_sub_list).to(device),
        "c_title": torch.LongTensor(c_title_list).to(device),
        "c_title_mask": torch.LongTensor(c_title_mask_list).to(device),
        "c_abs": torch.LongTensor(c_abs_list).to(device),
        "c_abs_mask": torch.LongTensor(c_abs_mask_list).to(device),
        "c_cat": torch.LongTensor(c_cat_list).to(device),
        "c_sub": torch.LongTensor(c_sub_list).to(device),
        "labels": torch.LongTensor(label_list).to(device)
    }
def prepare_eval_batch(batch_df, news_df, news_index):
    h_title_list, h_title_mask_list = [], []
    h_abs_list, h_abs_mask_list = [], []
    h_cat_list, h_sub_list = [], []
    c_title_list, c_title_mask_list = [], []
    c_abs_list, c_abs_mask_list = [], []
    c_cat_list, c_sub_list = [], []
    label_list = []
    for _, row in batch_df.iterrows():
        hist = row["history_parsed"][-HISTORY_LEN:]
        imps = row["impressions_parsed"]
        if len(hist) == 0 or len(imps) == 0: continue
        h_t, h_tm, h_a, h_am, h_c, h_s = [], [], [], [], [], []
        for nid in hist:
            if nid in news_index:
                idx = news_index[nid]
                h_t.append(news_df.iloc[idx]["title_ids"])
                h_tm.append(news_df.iloc[idx]["title_mask"])
                h_a.append(news_df.iloc[idx]["abstract_ids"])
                h_am.append(news_df.iloc[idx]["abstract_mask"])
                h_c.append(news_df.iloc[idx]["cat_id"])
                h_s.append(news_df.iloc[idx]["subcat_id"])
        while len(h_t) < HISTORY_LEN:
            h_t.append([0]*MAX_TITLE_LEN)
            h_tm.append([0]*MAX_TITLE_LEN)
            h_a.append([0]*MAX_ABS_LEN)
            h_am.append([0]*MAX_ABS_LEN)
            h_c.append(0)
            h_s.append(0)
        c_t, c_tm, c_a, c_am, c_c, c_s = [], [], [], [], [], []
        labels = []
        for nid, lbl in imps:
            if nid in news_index:
                idx = news_index[nid]
                c_t.append(news_df.iloc[idx]["title_ids"])
                c_tm.append(news_df.iloc[idx]["title_mask"])
                c_a.append(news_df.iloc[idx]["abstract_ids"])
                c_am.append(news_df.iloc[idx]["abstract_mask"])
                c_c.append(news_df.iloc[idx]["cat_id"])
                c_s.append(news_df.iloc[idx]["subcat_id"])
                labels.append(lbl)
        if len(c_t) == 0: continue
        h_title_list.append(h_t)
        h_title_mask_list.append(h_tm)
        h_abs_list.append(h_a)
        h_abs_mask_list.append(h_am)
        h_cat_list.append(h_c)
        h_sub_list.append(h_s)
        c_title_list.append(c_t)
        c_title_mask_list.append(c_tm)
        c_abs_list.append(c_a)
        c_abs_mask_list.append(c_am)
        c_cat_list.append(c_c)
        c_sub_list.append(c_s)
        label_list.append(labels)
    if not h_title_list: return None
    max_cands = max(len(lst) for lst in c_title_list)
    for i in range(len(c_title_list)):
        pad_len = max_cands - len(c_title_list[i])
        if pad_len > 0:
            c_title_list[i] += [[0] * MAX_TITLE_LEN] * pad_len
            c_title_mask_list[i] += [[0] * MAX_TITLE_LEN] * pad_len
            c_abs_list[i] += [[0] * MAX_ABS_LEN] * pad_len
            c_abs_mask_list[i] += [[0] * MAX_ABS_LEN] * pad_len
            c_cat_list[i] += [0] * pad_len
            c_sub_list[i] += [0] * pad_len
    return {
        "h_title": torch.LongTensor(h_title_list).to(device),
        "h_title_mask": torch.LongTensor(h_title_mask_list).to(device),
        "h_abs": torch.LongTensor(h_abs_list).to(device),
        "h_abs_mask": torch.LongTensor(h_abs_mask_list).to(device),
        "h_cat": torch.LongTensor(h_cat_list).to(device),
        "h_sub": torch.LongTensor(h_sub_list).to(device),
        "c_title": torch.LongTensor(c_title_list).to(device),
        "c_title_mask": torch.LongTensor(c_title_mask_list).to(device),
        "c_abs": torch.LongTensor(c_abs_list).to(device),
        "c_abs_mask": torch.LongTensor(c_abs_mask_list).to(device),
        "c_cat": torch.LongTensor(c_cat_list).to(device),
        "c_sub": torch.LongTensor(c_sub_list).to(device),
        "labels": label_list
    }
# ========================= EVALUATE =========================
@torch.no_grad()
def evaluate(df_split):
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    print(f"Evaluating {len(df_split):,} impressions with EVAL_BATCH_SIZE = {EVAL_BATCH_SIZE}...")
    for i in tqdm(range(0, len(df_split), EVAL_BATCH_SIZE), desc="Evaluating"):
        torch.cuda.empty_cache()
        gc.collect()
        batch_df = df_split.iloc[i:i + EVAL_BATCH_SIZE]
        batch = prepare_eval_batch(batch_df, news_df, news_index)
        if batch is None:
            continue
        scores = model(batch["h_title"], batch["h_title_mask"], batch["h_abs"], batch["h_abs_mask"],
                       batch["h_cat"], batch["h_sub"],
                       batch["c_title"], batch["c_title_mask"], batch["c_abs"], batch["c_abs_mask"],
                       batch["c_cat"], batch["c_sub"])
        torch.cuda.empty_cache()
        gc.collect()
        for j in range(len(batch["labels"])):
            true_labels = np.array(batch["labels"][j])
            num_real_cands = len(true_labels)
            score_vals = scores[j, :num_real_cands].cpu().numpy()
            if len(set(true_labels)) < 2:
                continue
            aucs.append(roc_auc_score(true_labels, score_vals))
            ranks = np.argsort(-score_vals)
            first_pos = np.where(true_labels[ranks] == 1)[0]
            if len(first_pos) > 0:
                mrrs.append(1.0 / (first_pos[0] + 1))
            def dcg(r, k):
                r = np.asarray(r)[:k]
                return np.sum(r / np.log2(np.arange(2, len(r) + 2)))
            def ndcg(r, k):
                dcg_val = dcg(r, k)
                idcg = dcg(sorted(r, reverse=True), k)
                return dcg_val / idcg if idcg > 0 else 0.0
            rel = true_labels[ranks]
            ndcg5s.append(ndcg(rel, 5))
            ndcg10s.append(ndcg(rel, 10))
    return {
        "AUC": np.mean(aucs) if aucs else 0,
        "MRR": np.mean(mrrs) if mrrs else 0,
        "nDCG@5": np.mean(ndcg5s) if ndcg5s else 0,
        "nDCG@10": np.mean(ndcg10s) if ndcg10s else 0
    }
# ========================= TRAINING =========================
def train_epoch():
    model.train()
    total_loss = 0.0
    torch.cuda.empty_cache()
    gc.collect()
    for i in tqdm(range(0, len(train_df), BATCH_SIZE), desc="Training"):
        batch_df = train_df.iloc[i:i + BATCH_SIZE]
        batch = prepare_train_batch(batch_df, news_df, news_index)
        if batch is None: continue
        with torch.cuda.amp.autocast():
            scores = model(batch["h_title"], batch["h_title_mask"], batch["h_abs"], batch["h_abs_mask"],
                           batch["h_cat"], batch["h_sub"],
                           batch["c_title"], batch["c_title_mask"], batch["c_abs"], batch["c_abs_mask"],
                           batch["c_cat"], batch["c_sub"])
            loss = naml_loss(scores, batch["labels"])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / (len(train_df) // BATCH_SIZE + 1)
# ========================= TRAINING LOOP - RESUME =========================
start_epoch = 5
best_auc = 0.0
print("\n" + "="*80)
print(f"RESUME TRAINING TỪ EPOCH {start_epoch}")
print("="*80)
for epoch in range(start_epoch, EPOCHS + 1):
    print(f"\n=== EPOCH {epoch}/{EPOCHS} ===")
    train_loss = train_epoch()
    print(f"Train Loss: {train_loss:.4f}")
    val_metrics = evaluate(val_df)
    print(f"Val → AUC: {val_metrics['AUC']:.4f} | MRR: {val_metrics['MRR']:.4f} | "
          f"nDCG@5: {val_metrics['nDCG@5']:.4f} | nDCG@10: {val_metrics['nDCG@10']:.4f}")
    if val_metrics["AUC"] > best_auc:
        best_auc = val_metrics["AUC"]
        torch.save(model.state_dict(), "/kaggle/working/naml_distilbert_best.pt")
        print("Saved best model")
# ========================= SAU KHI TRAIN EPOCH 5 XONG → LƯU VÀO /kaggle/working/ =========================
final_save_path = "/kaggle/working/naml_distilbert_best.pt"
torch.save(model.state_dict(), final_save_path)
print(f"\n✅ ĐÃ LƯU MODEL SAU KHI TRAIN EPOCH 5 XONG VÀO: {final_save_path}")
print("\n=== FINAL TEST EVALUATION ===")
test_metrics = evaluate(test_df)
print(f"Test → AUC: {test_metrics['AUC']:.4f} | MRR: {test_metrics['MRR']:.4f} | "
      f"nDCG@5: {test_metrics['nDCG@5']:.4f} | nDCG@10: {test_metrics['nDCG@10']:.4f}")
print(f"\nBest Validation AUC: {best_auc:.4f}")
print("HOÀN TẤT!")