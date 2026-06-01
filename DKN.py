import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random
from collections import defaultdict

# ====================== SETTINGS & SEED ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50
WORD_EMB_DIM = 100
ENTITY_EMB_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]
BATCH_SIZE = 128
EPOCHS = 5

# ====================== DATA LOADING ======================
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings

def preprocess_mind(news_path, behavior_path, entity_emb_dict):
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    all_titles = news_df['Title'].fillna("").tolist()
    word_dict = {"[PAD]": 0, "[UNK]": 1}
    for title in all_titles:
        for word in title.lower().split():
            if word not in word_dict: word_dict[word] = len(word_dict)

    entity_dict = {"[PAD]": 0, "[UNK]": 1}
    for eid in entity_emb_dict.keys(): entity_dict[eid] = len(entity_dict)

    def get_news_features(row):
        words = row['Title'].lower().split()[:MAX_TITLE_LEN]
        w_idx = [word_dict.get(w, 1) for w in words] + [0]*(MAX_TITLE_LEN - len(words))
        import json
        try:
            ents = json.loads(row['TitleEntities'])
            e_list = [entity_dict.get(e['EntityID'], 1) for e in ents][:MAX_TITLE_LEN]
        except: e_list = []
        e_idx = e_list + [0]*(MAX_TITLE_LEN - len(e_list))
        return w_idx, e_idx

    news_features = {row['NewsID']: get_news_features(row) for _, row in news_df.iterrows()}
    behaviors = pd.read_csv(behavior_path, sep='\t', header=None, names=["ImpID", "UID", "Time", "History", "Impressions"])
    return news_features, behaviors, word_dict, entity_dict

# ====================== METRICS ======================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

# ====================== MODEL ======================
class KCNN(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.word_embedding = nn.Embedding(word_num, config['w_dim'], padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_num, config['e_dim'], padding_idx=0)
        self.transform_matrix = nn.Parameter(torch.empty(config['e_dim'], config['w_dim']).uniform_(-0.1, 0.1))
        self.convs = nn.ModuleList([nn.Conv2d(2, config['n_filters'], (w, config['w_dim'])) for w in config['windows']])

    def forward(self, words, entities):
        w_vec = self.word_embedding(words)
        e_vec = self.entity_embedding(entities)
        e_transformed = torch.tanh(torch.matmul(e_vec, self.transform_matrix))
        x = torch.stack([w_vec, e_transformed], dim=1)
        pooled = [F.relu(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)

class DKNModel(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.encoder = KCNN(word_num, entity_num, config)
        feat_dim = len(config['windows']) * config['n_filters']
        self.attn = nn.Sequential(nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, hw, he, cw, ce):
        B, H, L = hw.shape
        c_vec = self.encoder(cw, ce)
        h_vecs = self.encoder(hw.view(-1, L), he.view(-1, L)).view(B, H, -1)
        c_rep = c_vec.unsqueeze(1).expand(-1, H, -1)
        attn_w = F.softmax(self.attn(torch.cat([c_rep, h_vecs], dim=-1)).squeeze(-1), dim=1)
        u_vec = torch.bmm(attn_w.unsqueeze(1), h_vecs).squeeze(1)
        return self.fc(u_vec + c_vec).squeeze(-1)

# ====================== DATASET ======================
class MINDDataset(Dataset):
    def __init__(self, behaviors, news_feat, mode='train'):
        self.data, self.news_feat = [], news_feat
        for _, row in behaviors.iterrows():
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            if mode == 'train':
                for imp in str(row['Impressions']).split():
                    nid, lab = imp.split('-'); self.data.append((hist, nid, int(lab)))
            else:
                imps = str(row['Impressions']).split()
                self.data.append((hist, [x.split('-')[0] for x in imps], [int(x.split('-')[1]) for x in imps]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h_ids, c_id, label = self.data[idx]
        hw, he = [], []
        for nid in h_ids:
            w, e = self.news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            hw.append(w); he.append(e)
        while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
        if isinstance(c_id, str):
            cw, ce = self.news_feat.get(c_id, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cw), torch.tensor(ce), torch.tensor(label, dtype=torch.float32)
        else:
            cws, ces = [], []
            for cid in c_id:
                w, e = self.news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                cws.append(w); ces.append(e)
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cws), torch.tensor(ces), torch.tensor(label, dtype=torch.float32)

# ====================== EXECUTION ======================
ent_emb = load_embeddings('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/entity_embedding.vec')
news_feat, train_behav, w_dict, e_dict = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv', ent_emb)
_, val_behav, _, _ = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/behaviors.tsv', ent_emb)

train_loader = DataLoader(MINDDataset(train_behav, news_feat), batch_size=BATCH_SIZE, shuffle=True)
model = DKNModel(len(w_dict), len(e_dict), {'w_dim':100, 'e_dim':100, 'n_filters':100, 'windows':[1,2,3]}).to(device)
with torch.no_grad():
    for eid, idx in e_dict.items():
        if eid in ent_emb: model.encoder.entity_embedding.weight[idx] = torch.tensor(ent_emb[eid])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, behaviors, news_feat):
    model.eval()
    m = {'auc':[], 'mrr':[], 'ndcg5':[], 'ndcg10':[]}
    with torch.no_grad():
        # Process ALL behaviors in validation set instead of sampling
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Evaluating"):
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            hw, he = [], []
            for nid in hist:
                w, e = news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                hw.append(w); he.append(e)
            while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
            imps = str(row['Impressions']).split(); labs, scs = [], []
            for imp in imps:
                cid, lab = imp.split('-')
                cw, ce = news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                s = model(torch.tensor([hw]).to(device), torch.tensor([he]).to(device), torch.tensor([cw]).to(device), torch.tensor([ce]).to(device))
                scs.append(torch.sigmoid(s).item()); labs.append(int(lab))
            if len(set(labs)) > 1:
                m['auc'].append(roc_auc_score(labs, scs))
                m['mrr'].append(mrr_score(np.array(labs), np.array(scs)))
                m['ndcg5'].append(ndcg_score(np.array(labs), np.array(scs), 5))
                m['ndcg10'].append(ndcg_score(np.array(labs), np.array(scs), 10))
    return {k: np.mean(v) for k, v in m.items()}

for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for hw, he, cw, ce, lb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        hw, he, cw, ce, lb = hw.to(device), he.to(device), cw.to(device), ce.to(device), lb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(hw, he, cw, ce), lb)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    res = evaluate(model, val_behav, news_feat)
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {res['auc']:.4f} | MRR: {res['mrr']:.4f} | nDCG@5: {res['ndcg5']:.4f} | nDCG@10: {res['ndcg10']:.4f}")
    # Save the model after each epoch
    torch.save(model.state_dict(), f"dkn_model_epoch_{epoch+1}.pth")
    print(f"Model saved as dkn_model_epoch_{epoch+1}.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random
from collections import defaultdict
# ====================== SETTINGS & SEED ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50
WORD_EMB_DIM = 100
ENTITY_EMB_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]
BATCH_SIZE = 128
EPOCHS = 5
# ====================== DATA LOADING ======================
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings
def preprocess_mind(news_path, behavior_path, entity_emb_dict):
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    all_titles = news_df['Title'].fillna("").tolist()
    word_dict = {"[PAD]": 0, "[UNK]": 1}
    for title in all_titles:
        for word in title.lower().split():
            if word not in word_dict: word_dict[word] = len(word_dict)
    entity_dict = {"[PAD]": 0, "[UNK]": 1}
    for eid in entity_emb_dict.keys(): entity_dict[eid] = len(entity_dict)
    def get_news_features(row):
        words = row['Title'].lower().split()[:MAX_TITLE_LEN]
        w_idx = [word_dict.get(w, 1) for w in words] + [0]*(MAX_TITLE_LEN - len(words))
        import json
        try:
            ents = json.loads(row['TitleEntities'])
            e_list = [entity_dict.get(e['EntityID'], 1) for e in ents][:MAX_TITLE_LEN]
        except: e_list = []
        e_idx = e_list + [0]*(MAX_TITLE_LEN - len(e_list))
        return w_idx, e_idx
    news_features = {row['NewsID']: get_news_features(row) for _, row in news_df.iterrows()}
    behaviors = pd.read_csv(behavior_path, sep='\t', header=None, names=["ImpID", "UID", "Time", "History", "Impressions"])
    return news_features, behaviors, word_dict, entity_dict
# ====================== METRICS ======================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0
def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
# ====================== MODEL ======================
class KCNN(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.word_embedding = nn.Embedding(word_num, config['w_dim'], padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_num, config['e_dim'], padding_idx=0)
        self.transform_matrix = nn.Parameter(torch.empty(config['e_dim'], config['w_dim']).uniform_(-0.1, 0.1))
        self.convs = nn.ModuleList([nn.Conv2d(2, config['n_filters'], (w, config['w_dim'])) for w in config['windows']])
    def forward(self, words, entities):
        w_vec = self.word_embedding(words)
        e_vec = self.entity_embedding(entities)
        e_transformed = torch.tanh(torch.matmul(e_vec, self.transform_matrix))
        x = torch.stack([w_vec, e_transformed], dim=1)
        pooled = [F.relu(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)
class DKNModel(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.encoder = KCNN(word_num, entity_num, config)
        feat_dim = len(config['windows']) * config['n_filters']
        self.attn = nn.Sequential(nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, hw, he, cw, ce):
        B, H, L = hw.shape
        c_vec = self.encoder(cw, ce)
        h_vecs = self.encoder(hw.view(-1, L), he.view(-1, L)).view(B, H, -1)
        c_rep = c_vec.unsqueeze(1).expand(-1, H, -1)
        attn_w = F.softmax(self.attn(torch.cat([c_rep, h_vecs], dim=-1)).squeeze(-1), dim=1)
        u_vec = torch.bmm(attn_w.unsqueeze(1), h_vecs).squeeze(1)
        return self.fc(u_vec + c_vec).squeeze(-1)
# ====================== DATASET ======================
class MINDDataset(Dataset):
    def __init__(self, behaviors, news_feat, mode='train'):
        self.data, self.news_feat = [], news_feat
        for _, row in behaviors.iterrows():
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            if mode == 'train':
                for imp in str(row['Impressions']).split():
                    nid, lab = imp.split('-'); self.data.append((hist, nid, int(lab)))
            else:
                imps = str(row['Impressions']).split()
                self.data.append((hist, [x.split('-')[0] for x in imps], [int(x.split('-')[1]) for x in imps]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h_ids, c_id, label = self.data[idx]
        hw, he = [], []
        for nid in h_ids:
            w, e = self.news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            hw.append(w); he.append(e)
        while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
        if isinstance(c_id, str):
            cw, ce = self.news_feat.get(c_id, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cw), torch.tensor(ce), torch.tensor(label, dtype=torch.float32)
        else:
            cws, ces = [], []
            for cid in c_id:
                w, e = self.news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                cws.append(w); ces.append(e)
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cws), torch.tensor(ces), torch.tensor(label, dtype=torch.float32)
# ====================== EXECUTION ======================
ent_emb = load_embeddings('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/entity_embedding.vec')
news_feat, train_behav, w_dict, e_dict = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv', ent_emb)
_, val_behav, _, _ = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/behaviors.tsv', ent_emb)
train_loader = DataLoader(MINDDataset(train_behav, news_feat), batch_size=BATCH_SIZE, shuffle=True)
model = DKNModel(len(w_dict), len(e_dict), {'w_dim':100, 'e_dim':100, 'n_filters':100, 'windows':[1,2,3]}).to(device)
with torch.no_grad():
    for eid, idx in e_dict.items():
        if eid in ent_emb: model.encoder.entity_embedding.weight[idx] = torch.tensor(ent_emb[eid])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()
def evaluate(model, behaviors, news_feat):
    model.eval()
    m = {'auc':[], 'mrr':[], 'ndcg5':[], 'ndcg10':[]}
    with torch.no_grad():
        # Process ALL behaviors in validation set instead of sampling
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Evaluating"):
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            hw, he = [], []
            for nid in hist:
                w, e = news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                hw.append(w); he.append(e)
            while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
            imps = str(row['Impressions']).split(); labs, scs = [], []
            for imp in imps:
                cid, lab = imp.split('-')
                cw, ce = news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                s = model(torch.tensor([hw]).to(device), torch.tensor([he]).to(device), torch.tensor([cw]).to(device), torch.tensor([ce]).to(device))
                scs.append(torch.sigmoid(s).item()); labs.append(int(lab))
            if len(set(labs)) > 1:
                m['auc'].append(roc_auc_score(labs, scs))
                m['mrr'].append(mrr_score(np.array(labs), np.array(scs)))
                m['ndcg5'].append(ndcg_score(np.array(labs), np.array(scs), 5))
                m['ndcg10'].append(ndcg_score(np.array(labs), np.array(scs), 10))
    return {k: np.mean(v) for k, v in m.items()}
for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for hw, he, cw, ce, lb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        hw, he, cw, ce, lb = hw.to(device), he.to(device), cw.to(device), ce.to(device), lb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(hw, he, cw, ce), lb)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    res = evaluate(model, val_behav, news_feat)
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {res['auc']:.4f} | MRR: {res['mrr']:.4f} | nDCG@5: {res['ndcg5']:.4f} | nDCG@10: {res['ndcg10']:.4f}")
    # Save the model after each epoch - LƯU VÀO /kaggle/working/
    torch.save(model.state_dict(), f"/kaggle/working/dkn_model_epoch_{epoch+1}.pth")
    print(f"Model saved as /kaggle/working/dkn_model_epoch_{epoch+1}.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random
from collections import defaultdict

# ====================== SETTINGS & SEED ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50
WORD_EMB_DIM = 100
ENTITY_EMB_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]
BATCH_SIZE = 128
EPOCHS = 5  # total epochs = 5, we will start from epoch 4 (index 3)

# ====================== DATA LOADING ======================
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings

def preprocess_mind(news_path, behavior_path, entity_emb_dict):
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    all_titles = news_df['Title'].fillna("").tolist()
    word_dict = {"[PAD]": 0, "[UNK]": 1}
    for title in all_titles:
        for word in title.lower().split():
            if word not in word_dict: word_dict[word] = len(word_dict)
    entity_dict = {"[PAD]": 0, "[UNK]": 1}
    for eid in entity_emb_dict.keys(): entity_dict[eid] = len(entity_dict)
    
    def get_news_features(row):
        words = row['Title'].lower().split()[:MAX_TITLE_LEN]
        w_idx = [word_dict.get(w, 1) for w in words] + [0]*(MAX_TITLE_LEN - len(words))
        import json
        try:
            ents = json.loads(row['TitleEntities'])
            e_list = [entity_dict.get(e['EntityID'], 1) for e in ents][:MAX_TITLE_LEN]
        except: e_list = []
        e_idx = e_list + [0]*(MAX_TITLE_LEN - len(e_list))
        return w_idx, e_idx
    
    news_features = {row['NewsID']: get_news_features(row) for _, row in news_df.iterrows()}
    behaviors = pd.read_csv(behavior_path, sep='\t', header=None, names=["ImpID", "UID", "Time", "History", "Impressions"])
    return news_features, behaviors, word_dict, entity_dict

# ====================== METRICS ======================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

# ====================== MODEL ======================
class KCNN(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.word_embedding = nn.Embedding(word_num, config['w_dim'], padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_num, config['e_dim'], padding_idx=0)
        self.transform_matrix = nn.Parameter(torch.empty(config['e_dim'], config['w_dim']).uniform_(-0.1, 0.1))
        self.convs = nn.ModuleList([nn.Conv2d(2, config['n_filters'], (w, config['w_dim'])) for w in config['windows']])
    
    def forward(self, words, entities):
        w_vec = self.word_embedding(words)
        e_vec = self.entity_embedding(entities)
        e_transformed = torch.tanh(torch.matmul(e_vec, self.transform_matrix))
        x = torch.stack([w_vec, e_transformed], dim=1)
        pooled = [F.relu(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)

class DKNModel(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.encoder = KCNN(word_num, entity_num, config)
        feat_dim = len(config['windows']) * config['n_filters']
        self.attn = nn.Sequential(nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc = nn.Linear(feat_dim, 1)
    
    def forward(self, hw, he, cw, ce):
        B, H, L = hw.shape
        c_vec = self.encoder(cw, ce)
        h_vecs = self.encoder(hw.view(-1, L), he.view(-1, L)).view(B, H, -1)
        c_rep = c_vec.unsqueeze(1).expand(-1, H, -1)
        attn_w = F.softmax(self.attn(torch.cat([c_rep, h_vecs], dim=-1)).squeeze(-1), dim=1)
        u_vec = torch.bmm(attn_w.unsqueeze(1), h_vecs).squeeze(1)
        return self.fc(u_vec + c_vec).squeeze(-1)

# ====================== DATASET ======================
class MINDDataset(Dataset):
    def __init__(self, behaviors, news_feat, mode='train'):
        self.data, self.news_feat = [], news_feat
        for _, row in behaviors.iterrows():
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            if mode == 'train':
                for imp in str(row['Impressions']).split():
                    nid, lab = imp.split('-'); self.data.append((hist, nid, int(lab)))
            else:
                imps = str(row['Impressions']).split()
                self.data.append((hist, [x.split('-')[0] for x in imps], [int(x.split('-')[1]) for x in imps]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        h_ids, c_id, label = self.data[idx]
        hw, he = [], []
        for nid in h_ids:
            w, e = self.news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            hw.append(w); he.append(e)
        while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
        if isinstance(c_id, str):
            cw, ce = self.news_feat.get(c_id, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cw), torch.tensor(ce), torch.tensor(label, dtype=torch.float32)
        else:
            cws, ces = [], []
            for cid in c_id:
                w, e = self.news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                cws.append(w); ces.append(e)
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cws), torch.tensor(ces), torch.tensor(label, dtype=torch.float32)

# ====================== EXECUTION ======================
ent_emb = load_embeddings('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/entity_embedding.vec')
news_feat, train_behav, w_dict, e_dict = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv', ent_emb)
_, val_behav, _, _ = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/behaviors.tsv', ent_emb)

train_loader = DataLoader(MINDDataset(train_behav, news_feat), batch_size=BATCH_SIZE, shuffle=True)

model = DKNModel(len(w_dict), len(e_dict), {'w_dim':100, 'e_dim':100, 'n_filters':100, 'windows':[1,2,3]}).to(device)

# Load pre-trained weights from epoch 3 (nếu có)
checkpoint_path = '/kaggle/input/datasets/neyugnhieu/dkn-mode-save/dkn_model_epoch_3.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")
else:
    print(f"Checkpoint not found at {checkpoint_path}, training from scratch.")
    # Chỉ khởi tạo entity embeddings khi train từ đầu (không có checkpoint)
    with torch.no_grad():
        for eid, idx in e_dict.items():
            if eid in ent_emb:
                model.encoder.entity_embedding.weight[idx] = torch.tensor(ent_emb[eid])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, behaviors, news_feat):
    model.eval()
    m = {'auc':[], 'mrr':[], 'ndcg5':[], 'ndcg10':[]}
    with torch.no_grad():
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Evaluating"):
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            hw, he = [], []
            for nid in hist:
                w, e = news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                hw.append(w); he.append(e)
            while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
            imps = str(row['Impressions']).split(); labs, scs = [], []
            for imp in imps:
                cid, lab = imp.split('-')
                cw, ce = news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                s = model(torch.tensor([hw]).to(device), torch.tensor([he]).to(device), torch.tensor([cw]).to(device), torch.tensor([ce]).to(device))
                scs.append(torch.sigmoid(s).item()); labs.append(int(lab))
            if len(set(labs)) > 1:
                m['auc'].append(roc_auc_score(labs, scs))
                m['mrr'].append(mrr_score(np.array(labs), np.array(scs)))
                m['ndcg5'].append(ndcg_score(np.array(labs), np.array(scs), 5))
                m['ndcg10'].append(ndcg_score(np.array(labs), np.array(scs), 10))
    return {k: np.mean(v) for k, v in m.items()}

# Continue training from epoch 4 to 5 (total EPOCHS=5, start_epoch=3 means we already did 3)
start_epoch = 3  # model đã được train 3 epoch
for epoch in range(start_epoch, EPOCHS):
    model.train(); total_loss = 0
    for hw, he, cw, ce, lb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        hw, he, cw, ce, lb = hw.to(device), he.to(device), cw.to(device), ce.to(device), lb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(hw, he, cw, ce), lb)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    
    res = evaluate(model, val_behav, news_feat)
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {res['auc']:.4f} | MRR: {res['mrr']:.4f} | nDCG@5: {res['ndcg5']:.4f} | nDCG@10: {res['ndcg10']:.4f}")
    
    # Save the model after each epoch
    torch.save(model.state_dict(), f"/kaggle/working/dkn_model_epoch_{epoch+1}.pth")
    print(f"Model saved as /kaggle/working/dkn_model_epoch_{epoch+1}.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random
from collections import defaultdict

# ====================== SETTINGS & SEED ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50
WORD_EMB_DIM = 100
ENTITY_EMB_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]
BATCH_SIZE = 128
EPOCHS = 5

# ====================== DATA LOADING ======================
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings

def preprocess_mind(news_path, behavior_path, entity_emb_dict):
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    all_titles = news_df['Title'].fillna("").tolist()
    word_dict = {"[PAD]": 0, "[UNK]": 1}
    for title in all_titles:
        for word in title.lower().split():
            if word not in word_dict: word_dict[word] = len(word_dict)

    entity_dict = {"[PAD]": 0, "[UNK]": 1}
    for eid in entity_emb_dict.keys(): entity_dict[eid] = len(entity_dict)

    def get_news_features(row):
        words = row['Title'].lower().split()[:MAX_TITLE_LEN]
        w_idx = [word_dict.get(w, 1) for w in words] + [0]*(MAX_TITLE_LEN - len(words))
        import json
        try:
            ents = json.loads(row['TitleEntities'])
            e_list = [entity_dict.get(e['EntityID'], 1) for e in ents][:MAX_TITLE_LEN]
        except: e_list = []
        e_idx = e_list + [0]*(MAX_TITLE_LEN - len(e_list))
        return w_idx, e_idx

    news_features = {row['NewsID']: get_news_features(row) for _, row in news_df.iterrows()}
    behaviors = pd.read_csv(behavior_path, sep='\t', header=None, names=["ImpID", "UID", "Time", "History", "Impressions"])
    return news_features, behaviors, word_dict, entity_dict

# ====================== METRICS ======================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

# ====================== MODEL ======================
class KCNN(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.word_embedding = nn.Embedding(word_num, config['w_dim'], padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_num, config['e_dim'], padding_idx=0)
        self.transform_matrix = nn.Parameter(torch.empty(config['e_dim'], config['w_dim']).uniform_(-0.1, 0.1))
        self.convs = nn.ModuleList([nn.Conv2d(2, config['n_filters'], (w, config['w_dim'])) for w in config['windows']])

    def forward(self, words, entities):
        w_vec = self.word_embedding(words)
        e_vec = self.entity_embedding(entities)
        e_transformed = torch.tanh(torch.matmul(e_vec, self.transform_matrix))
        x = torch.stack([w_vec, e_transformed], dim=1)
        pooled = [F.relu(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)

class DKNModel(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.encoder = KCNN(word_num, entity_num, config)
        feat_dim = len(config['windows']) * config['n_filters']
        self.attn = nn.Sequential(nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, hw, he, cw, ce):
        B, H, L = hw.shape
        c_vec = self.encoder(cw, ce)
        h_vecs = self.encoder(hw.view(-1, L), he.view(-1, L)).view(B, H, -1)
        c_rep = c_vec.unsqueeze(1).expand(-1, H, -1)
        attn_w = F.softmax(self.attn(torch.cat([c_rep, h_vecs], dim=-1)).squeeze(-1), dim=1)
        u_vec = torch.bmm(attn_w.unsqueeze(1), h_vecs).squeeze(1)
        return self.fc(u_vec + c_vec).squeeze(-1)

# ====================== DATASET ======================
class MINDDataset(Dataset):
    def __init__(self, behaviors, news_feat, mode='train'):
        self.data, self.news_feat = [], news_feat
        for _, row in behaviors.iterrows():
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            if mode == 'train':
                for imp in str(row['Impressions']).split():
                    nid, lab = imp.split('-'); self.data.append((hist, nid, int(lab)))
            else:
                imps = str(row['Impressions']).split()
                self.data.append((hist, [x.split('-')[0] for x in imps], [int(x.split('-')[1]) for x in imps]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h_ids, c_id, label = self.data[idx]
        hw, he = [], []
        for nid in h_ids:
            w, e = self.news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            hw.append(w); he.append(e)
        while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
        if isinstance(c_id, str):
            cw, ce = self.news_feat.get(c_id, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cw), torch.tensor(ce), torch.tensor(label, dtype=torch.float32)
        else:
            cws, ces = [], []
            for cid in c_id:
                w, e = self.news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                cws.append(w); ces.append(e)
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cws), torch.tensor(ces), torch.tensor(label, dtype=torch.float32)

# ====================== EVALUATION FUNCTION ======================
def evaluate(model, behaviors, news_feat):
    model.eval()
    m = {'auc':[], 'mrr':[], 'ndcg5':[], 'ndcg10':[]}
    with torch.no_grad():
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Evaluating"):
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            hw, he = [], []
            for nid in hist:
                w, e = news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                hw.append(w); he.append(e)
            while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
            imps = str(row['Impressions']).split()
            labs, scs = [], []
            for imp in imps:
                cid, lab = imp.split('-')
                cw, ce = news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                s = model(torch.tensor([hw]).to(device), torch.tensor([he]).to(device),
                          torch.tensor([cw]).to(device), torch.tensor([ce]).to(device))
                scs.append(torch.sigmoid(s).item())
                labs.append(int(lab))
            if len(set(labs)) > 1:
                m['auc'].append(roc_auc_score(labs, scs))
                m['mrr'].append(mrr_score(np.array(labs), np.array(scs)))
                m['ndcg5'].append(ndcg_score(np.array(labs), np.array(scs), 5))
                m['ndcg10'].append(ndcg_score(np.array(labs), np.array(scs), 10))
    return {k: np.mean(v) for k, v in m.items()}

# ====================== LOAD DATA & MODEL ======================
print("Loading entity embeddings...")
ent_emb = load_embeddings('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/entity_embedding.vec')

print("Preprocessing training data (to get word/entity dicts and news features)...")
news_feat, train_behav, w_dict, e_dict = preprocess_mind(
    '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/news.tsv',
    '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv',
    ent_emb
)

print("Preprocessing validation data...")
_, val_behav, _, _ = preprocess_mind(
    '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/news.tsv',
    '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/behaviors.tsv',
    ent_emb
)

# Khởi tạo model với cùng config
config = {'w_dim': 100, 'e_dim': 100, 'n_filters': 100, 'windows': [1, 2, 3]}
model = DKNModel(len(w_dict), len(e_dict), config).to(device)

# Load trọng số đã train (epoch 2)
model_path = '/kaggle/input/datasets/neyugnhieu/dkn-mode-save/dkn_model_epoch_2.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model loaded successfully!")

# ====================== EVALUATION ONLY ======================
print("\nStarting evaluation on validation set (used as test)...")
results = evaluate(model, val_behav, news_feat)
print("\n========== FINAL RESULTS ==========")
print(f"AUC:      {results['auc']:.4f}")
print(f"MRR:      {results['mrr']:.4f}")
print(f"nDCG@5:   {results['ndcg5']:.4f}")
print(f"nDCG@10:  {results['ndcg10']:.4f}")
print("===================================")

# ====================== SAVE BEST MODEL ======================
torch.save(model.state_dict(), "dkn_best_model.pt")
print("Model saved as dkn_best_model.pt")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random
from collections import defaultdict

# ====================== SETTINGS & SEED ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50
WORD_EMB_DIM = 100
ENTITY_EMB_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]
BATCH_SIZE = 128
EPOCHS = 5

# ====================== DATA LOADING ======================
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings

def preprocess_mind(news_path, behavior_path, entity_emb_dict):
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    all_titles = news_df['Title'].fillna("").tolist()
    word_dict = {"[PAD]": 0, "[UNK]": 1}
    for title in all_titles:
        for word in title.lower().split():
            if word not in word_dict: word_dict[word] = len(word_dict)

    entity_dict = {"[PAD]": 0, "[UNK]": 1}
    for eid in entity_emb_dict.keys(): entity_dict[eid] = len(entity_dict)

    def get_news_features(row):
        words = row['Title'].lower().split()[:MAX_TITLE_LEN]
        w_idx = [word_dict.get(w, 1) for w in words] + [0]*(MAX_TITLE_LEN - len(words))
        import json
        try:
            ents = json.loads(row['TitleEntities'])
            e_list = [entity_dict.get(e['EntityID'], 1) for e in ents][:MAX_TITLE_LEN]
        except: e_list = []
        e_idx = e_list + [0]*(MAX_TITLE_LEN - len(e_list))
        return w_idx, e_idx

    news_features = {row['NewsID']: get_news_features(row) for _, row in news_df.iterrows()}
    behaviors = pd.read_csv(behavior_path, sep='\t', header=None, names=["ImpID", "UID", "Time", "History", "Impressions"])
    return news_features, behaviors, word_dict, entity_dict

# ====================== METRICS ======================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

# ====================== MODEL ======================
class KCNN(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.word_embedding = nn.Embedding(word_num, config['w_dim'], padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_num, config['e_dim'], padding_idx=0)
        self.transform_matrix = nn.Parameter(torch.empty(config['e_dim'], config['w_dim']).uniform_(-0.1, 0.1))
        self.convs = nn.ModuleList([nn.Conv2d(2, config['n_filters'], (w, config['w_dim'])) for w in config['windows']])

    def forward(self, words, entities):
        w_vec = self.word_embedding(words)
        e_vec = self.entity_embedding(entities)
        e_transformed = torch.tanh(torch.matmul(e_vec, self.transform_matrix))
        x = torch.stack([w_vec, e_transformed], dim=1)
        pooled = [F.relu(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)

class DKNModel(nn.Module):
    def __init__(self, word_num, entity_num, config):
        super().__init__()
        self.encoder = KCNN(word_num, entity_num, config)
        feat_dim = len(config['windows']) * config['n_filters']
        self.attn = nn.Sequential(nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, hw, he, cw, ce):
        B, H, L = hw.shape
        c_vec = self.encoder(cw, ce)
        h_vecs = self.encoder(hw.view(-1, L), he.view(-1, L)).view(B, H, -1)
        c_rep = c_vec.unsqueeze(1).expand(-1, H, -1)
        attn_w = F.softmax(self.attn(torch.cat([c_rep, h_vecs], dim=-1)).squeeze(-1), dim=1)
        u_vec = torch.bmm(attn_w.unsqueeze(1), h_vecs).squeeze(1)
        return self.fc(u_vec + c_vec).squeeze(-1)

# ====================== DATASET ======================
class MINDDataset(Dataset):
    def __init__(self, behaviors, news_feat, mode='train'):
        self.data, self.news_feat = [], news_feat
        for _, row in behaviors.iterrows():
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            if mode == 'train':
                for imp in str(row['Impressions']).split():
                    nid, lab = imp.split('-'); self.data.append((hist, nid, int(lab)))
            else:
                imps = str(row['Impressions']).split()
                self.data.append((hist, [x.split('-')[0] for x in imps], [int(x.split('-')[1]) for x in imps]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h_ids, c_id, label = self.data[idx]
        hw, he = [], []
        for nid in h_ids:
            w, e = self.news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            hw.append(w); he.append(e)
        while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
        if isinstance(c_id, str):
            cw, ce = self.news_feat.get(c_id, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cw), torch.tensor(ce), torch.tensor(label, dtype=torch.float32)
        else:
            cws, ces = [], []
            for cid in c_id:
                w, e = self.news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                cws.append(w); ces.append(e)
            return torch.tensor(hw), torch.tensor(he), torch.tensor(cws), torch.tensor(ces), torch.tensor(label, dtype=torch.float32)

# ====================== EXECUTION ======================
ent_emb = load_embeddings('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/entity_embedding.vec')
news_feat, train_behav, w_dict, e_dict = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv', ent_emb)
_, val_behav, _, _ = preprocess_mind('/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/news.tsv', '/kaggle/input/datasets/neyugnhieu/dataset-mind-small/MINDsmall_val/MINDsmall_val/behaviors.tsv', ent_emb)

# ====================== GIỚI HẠN MẪU ĐỂ TEST NHANH ======================
train_behav = train_behav.head(1000)   # chỉ dùng 1000 dòng behaviors để train
val_behav = val_behav.head(200)       # chỉ dùng 200 dòng behaviors để validation
print(f"Training samples: {len(train_behav)} | Validation samples: {len(val_behav)}")
# ========================================================================

train_loader = DataLoader(MINDDataset(train_behav, news_feat), batch_size=BATCH_SIZE, shuffle=True)
model = DKNModel(len(w_dict), len(e_dict), {'w_dim':100, 'e_dim':100, 'n_filters':100, 'windows':[1,2,3]}).to(device)
with torch.no_grad():
    for eid, idx in e_dict.items():
        if eid in ent_emb: model.encoder.entity_embedding.weight[idx] = torch.tensor(ent_emb[eid])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, behaviors, news_feat):
    model.eval()
    m = {'auc':[], 'mrr':[], 'ndcg5':[], 'ndcg10':[]}
    with torch.no_grad():
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Evaluating"):
            hist = str(row['History']).split()[-MAX_HISTORY_LEN:]
            if not hist: continue
            hw, he = [], []
            for nid in hist:
                w, e = news_feat.get(nid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                hw.append(w); he.append(e)
            while len(hw) < MAX_HISTORY_LEN: hw.append([0]*MAX_TITLE_LEN); he.append([0]*MAX_TITLE_LEN)
            imps = str(row['Impressions']).split(); labs, scs = [], []
            for imp in imps:
                cid, lab = imp.split('-')
                cw, ce = news_feat.get(cid, ([0]*MAX_TITLE_LEN, [0]*MAX_TITLE_LEN))
                s = model(torch.tensor([hw]).to(device), torch.tensor([he]).to(device), torch.tensor([cw]).to(device), torch.tensor([ce]).to(device))
                scs.append(torch.sigmoid(s).item()); labs.append(int(lab))
            if len(set(labs)) > 1:
                m['auc'].append(roc_auc_score(labs, scs))
                m['mrr'].append(mrr_score(np.array(labs), np.array(scs)))
                m['ndcg5'].append(ndcg_score(np.array(labs), np.array(scs), 5))
                m['ndcg10'].append(ndcg_score(np.array(labs), np.array(scs), 10))
    return {k: np.mean(v) for k, v in m.items()}

for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for hw, he, cw, ce, lb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        hw, he, cw, ce, lb = hw.to(device), he.to(device), cw.to(device), ce.to(device), lb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(hw, he, cw, ce), lb)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    res = evaluate(model, val_behav, news_feat)
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {res['auc']:.4f} | MRR: {res['mrr']:.4f} | nDCG@5: {res['ndcg5']:.4f} | nDCG@10: {res['ndcg10']:.4f}")
    # Save the model after each epoch
    torch.save(model.state_dict(), f"dkn_model_epoch_{epoch+1}.pth")
    print(f"Model saved as dkn_model_epoch_{epoch+1}.pth")