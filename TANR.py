# ====================================================================
# TOÀN BỘ CODE HUẤN LUYỆN TANR TRÊN MIND-SMALL (ĐÃ SỬA METRIC)
# METRIC: AUC, MRR, nDCG@5, nDCG@10 - Theo đúng paper
# ====================================================================

# ---------- 1. Cài đặt thư viện và chuẩn bị môi trường ----------
!pip install -q torch nltk scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os, sys, csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)

# Clone repository để lấy module attention
!git clone -q https://github.com/MinhNguyenDS/Co-NAML-LSTUR.git
%cd Co-NAML-LSTUR
sys.path.append('./model/general/attention')
from additive import AdditiveAttention

# Thiết lập device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- 2. Đường dẫn và load dữ liệu ----------
BASE_PATH = "/kaggle/input/datasets/nguyenueih12211221/dataset-mind-small"
TRAIN_PATH = os.path.join(BASE_PATH, "MINDsmall_train")
VAL_PATH   = os.path.join(BASE_PATH, "MINDsmall_val")

news_columns = ["NewsID","Category","SubCategory","Title","Abstract","URL","TitleEntities","AbstractEntities"]
behaviors_columns = ["ImpressionID","UserID","Time","History","Impressions"]

news_train = pd.read_csv(os.path.join(TRAIN_PATH, "news.tsv"), sep="\t", names=news_columns, quoting=csv.QUOTE_NONE)
news_val   = pd.read_csv(os.path.join(VAL_PATH, "news.tsv"), sep="\t", names=news_columns, quoting=csv.QUOTE_NONE)
behaviors_train = pd.read_csv(os.path.join(TRAIN_PATH, "behaviors.tsv"), sep="\t", names=behaviors_columns, quoting=csv.QUOTE_NONE)
behaviors_val   = pd.read_csv(os.path.join(VAL_PATH, "behaviors.tsv"), sep="\t", names=behaviors_columns, quoting=csv.QUOTE_NONE)

print(f"News Train: {news_train.shape}, News Val: {news_val.shape}")
print(f"Behaviors Train: {behaviors_train.shape}, Behaviors Val: {behaviors_val.shape}")

# ---------- 3. Xây dựng vocabulary và mapping cho NewsID ----------
news_id_to_idx = {}
news_idx_to_data = {}
all_news = pd.concat([news_train, news_val])
for _, row in all_news.iterrows():
    nid = row['NewsID']
    if nid not in news_id_to_idx:
        news_id_to_idx[nid] = len(news_id_to_idx)
        news_idx_to_data[news_id_to_idx[nid]] = row

def build_vocab(df):
    w2i = {'[PAD]':0, '[UNK]':1}
    wc = {}
    for _, row in df.iterrows():
        for w in word_tokenize(str(row['Title']).lower()):
            wc[w] = wc.get(w,0)+1
        for w in word_tokenize(str(row['Abstract']).lower()):
            wc[w] = wc.get(w,0)+1
    for k,v in wc.items():
        if v>=1:
            w2i[k] = len(w2i)
    return w2i

print("Building vocabulary...")
word2idx = build_vocab(news_train)
num_words = len(word2idx)
print(f"Vocabulary size: {num_words}")

# Tham số cố định
num_words_title = 20
num_clicked_news_a_user = 30
negative_sampling_ratio = 3
word_embedding_dim = 100
num_filters = 300
num_categories = 18

# ---------- 4. Pretrained embedding (ngẫu nhiên) ----------
pretrained_word_embedding = np.random.uniform(-0.1, 0.1, (num_words, word_embedding_dim))
pretrained_word_embedding[0] = 0.0
pretrained_word_embedding = torch.tensor(pretrained_word_embedding, dtype=torch.float32)

# ---------- 5. Tạo cache nội dung tin tức ----------
def news_to_content(news_id, w2i):
    if news_id not in news_id_to_idx or news_id=='PAD':
        return {'title': [0]*num_words_title}
    idx = news_id_to_idx[news_id]
    row = news_idx_to_data[idx]
    try:
        words = word_tokenize(str(row['Title']).lower())
    except:
        words = []
    indices = [w2i.get(w,1) for w in words[:num_words_title]]
    indices += [0]*(num_words_title - len(indices))
    return {'title': indices}

print("Caching news content...")
news_cache = {}
for nid in tqdm(news_id_to_idx.keys(), desc="Caching"):
    news_cache[nid] = news_to_content(nid, word2idx)
news_cache['PAD'] = news_to_content('PAD', word2idx)

# ---------- 6. Parse behaviors cho TRAINING ----------
def parse_behaviors_train(df):
    """
    Parse behaviors cho training: tạo các cặp (1 positive + K negatives)
    """
    parsed = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parsing train"):
        hist_str = str(row['History'])
        clicked = hist_str.split()[:num_clicked_news_a_user] if hist_str and hist_str!='nan' else []
        clicked = ['PAD']*(num_clicked_news_a_user-len(clicked)) + clicked
        impr_str = str(row['Impressions'])
        if not impr_str or impr_str=='nan': continue
        imprs = impr_str.split()
        pos = [x for x in imprs if x.endswith('-1')]
        neg = [x for x in imprs if x.endswith('-0')]
        if not neg: continue
        for p in pos:
            pid = p.split('-')[0]
            neg_samples = list(np.random.choice(neg, size=negative_sampling_ratio, replace=True))
            cand = [pid] + [n.split('-')[0] for n in neg_samples]
            lbl = [1] + [0]*negative_sampling_ratio
            parsed.append({
                'user': row['UserID'],
                'clicked_news': clicked,
                'candidate_news': cand,
                'clicked': lbl
            })
    return pd.DataFrame(parsed)

# ---------- 7. Parse behaviors cho VALIDATION (GIỮ NGUYÊN TẤT CẢ CANDIDATES) ----------
def parse_behaviors_val(df):
    """
    Parse behaviors cho validation: giữ TẤT CẢ candidates của mỗi impression
    để đánh giá metric chính xác
    """
    parsed = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parsing val"):
        hist_str = str(row['History'])
        clicked = hist_str.split()[:num_clicked_news_a_user] if hist_str and hist_str!='nan' else []
        clicked = ['PAD']*(num_clicked_news_a_user-len(clicked)) + clicked
        
        impr_str = str(row['Impressions'])
        if not impr_str or impr_str=='nan': continue
        
        imprs = impr_str.split()
        
        # Lấy TẤT CẢ candidates (cả positive và negative)
        candidate_news = [x.split('-')[0] for x in imprs]
        candidate_labels = [int(x.split('-')[1]) for x in imprs]
        
        parsed.append({
            'user': row['UserID'],
            'impression_id': row['ImpressionID'],
            'clicked_news': clicked,
            'candidate_news': candidate_news,
            'clicked': candidate_labels
        })
    
    return pd.DataFrame(parsed)

train_df = parse_behaviors_train(behaviors_train)
val_df   = parse_behaviors_val(behaviors_val)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
print(f"Val candidate lengths: min={val_df['candidate_news'].apply(len).min()}, max={val_df['candidate_news'].apply(len).max()}, mean={val_df['candidate_news'].apply(len).mean():.1f}")

# ---------- 8. Dataset class ----------
class TANRDataset(Dataset):
    def __init__(self, df, news_cache):
        self.data = df.reset_index(drop=True)
        self.cache = news_cache
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clicked_titles = [self.cache.get(nid, self.cache['PAD'])['title'] for nid in row['clicked_news']]
        cand_titles = [self.cache.get(nid, self.cache['PAD'])['title'] for nid in row['candidate_news']]
        
        return {
            'clicked_title': clicked_titles,
            'candidate_title': cand_titles,
            'clicked': row['clicked']
        }

train_dataset = TANRDataset(train_df, news_cache)
val_dataset = TANRDataset(val_df, news_cache)

# ---------- 9. Custom collate function ----------
def collate_fn_train(batch):
    """
    Collate cho training: tất cả sample có cùng kích thước (4 candidates)
    """
    clicked = np.array([item['clicked_title'] for item in batch], dtype=np.int64)
    cand = np.array([item['candidate_title'] for item in batch], dtype=np.int64)
    labels = np.array([item['clicked'] for item in batch], dtype=np.int64)
    
    return {
        'clicked_title': torch.from_numpy(clicked),
        'candidate_title': torch.from_numpy(cand),
        'clicked': torch.from_numpy(labels)
    }

def collate_fn_val(batch):
    """
    Collate cho validation: mỗi sample có thể có số lượng candidates khác nhau
    Sử dụng padding để batch được
    """
    max_cands = max(len(item['candidate_title']) for item in batch)
    
    clicked_list = []
    cand_list = []
    labels_list = []
    masks_list = []
    
    for item in batch:
        clicked_list.append(item['clicked_title'])
        
        # Pad candidates
        cand = item['candidate_title']
        labels = item['clicked']
        num_cands = len(cand)
        
        # Padding
        pad_len = max_cands - num_cands
        cand_padded = cand + [[0]*20] * pad_len
        labels_padded = labels + [0] * pad_len
        mask = [1]*num_cands + [0]*pad_len
        
        cand_list.append(cand_padded)
        labels_list.append(labels_padded)
        masks_list.append(mask)
    
    return {
        'clicked_title': torch.from_numpy(np.array(clicked_list, dtype=np.int64)),
        'candidate_title': torch.from_numpy(np.array(cand_list, dtype=np.int64)),
        'clicked': torch.from_numpy(np.array(labels_list, dtype=np.int64)),
        'mask': torch.from_numpy(np.array(masks_list, dtype=np.float32))
    }

# ---------- 10. Mô hình TANR ----------
class SimpleNewsEncoder(nn.Module):
    def __init__(self, num_words, emb_dim, num_filters):
        super().__init__()
        self.embedding = nn.Embedding(num_words, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(pretrained_word_embedding)
        self.conv1d = nn.Conv1d(emb_dim, num_filters, kernel_size=3, padding=1)
        self.attn = AdditiveAttention(100, num_filters)
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.drop(self.embedding(x))
        x = x.transpose(1,2)
        x = F.relu(self.conv1d(x))
        x = x.transpose(1,2)
        return self.attn(x)

class SimpleTANR(nn.Module):
    def __init__(self, num_words, emb_dim, num_filters, num_cat):
        super().__init__()
        self.news_enc = SimpleNewsEncoder(num_words, emb_dim, num_filters)
        self.user_attn = AdditiveAttention(100, num_filters)
        self.topic_pred = nn.Linear(num_filters, num_cat)
    
    def forward(self, cand_title, clicked_title):
        bs, nc, nw = cand_title.shape
        _, ncl, _ = clicked_title.shape
        
        enc_cand = self.news_enc(cand_title.view(-1, nw)).view(bs, nc, -1)
        enc_clicked = self.news_enc(clicked_title.view(-1, nw)).view(bs, ncl, -1)
        user_vec = self.user_attn(enc_clicked)
        click_prob = torch.bmm(enc_cand, user_vec.unsqueeze(-1)).squeeze(-1)
        topic_pred = self.topic_pred(enc_clicked.mean(dim=1))
        
        return click_prob, topic_pred

model = SimpleTANR(num_words, word_embedding_dim, num_filters, num_categories).to(device)
print(model)

# ---------- 11. DataLoaders ----------
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, collate_fn=collate_fn_train)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                        num_workers=0, collate_fn=collate_fn_val)

criterion = nn.CrossEntropyLoss()
topic_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- 12. Hàm tính metric (đúng theo paper) ----------
def dcg_score(y_true, y_score, k=10):
    """Tính DCG@k - chỉ tính trên k items đầu tiên"""
    order = np.argsort(y_score)[::-1]  # Sắp xếp giảm dần theo score
    y_true = np.take(y_true, order[:k])  # Lấy k items có score cao nhất
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)  # log2(rank+2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Tính nDCG@k"""
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0

def mrr_score(y_true, y_score):
    """
    Tính MRR (Mean Reciprocal Rank)
    MRR = 1/rank của item positive đầu tiên
    """
    order = np.argsort(y_score)[::-1]  # Sắp xếp giảm dần
    y_true_sorted = np.take(y_true, order)
    
    # Tìm vị trí đầu tiên có label = 1
    try:
        rank = np.where(y_true_sorted == 1)[0][0] + 1  # +1 vì rank bắt đầu từ 1
        return 1.0 / rank
    except IndexError:
        return 0.0

def calculate_single_user_metrics(y_true, y_score):
    """
    Tính tất cả metrics cho một impression
    """
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    
    mrr = mrr_score(y_true, y_score)
    ndcg5 = ndcg_score(y_true, y_score, 5)
    ndcg10 = ndcg_score(y_true, y_score, 10)
    
    return auc, mrr, ndcg5, ndcg10

# ---------- 13. Hàm validate (ĐÃ SỬA) ----------
def validate(model, loader, device):
    """
    Đánh giá model trên validation set với tất cả candidates của mỗi impression
    """
    model.eval()
    all_aucs = []
    all_mrrs = []
    all_ndcg5s = []
    all_ndcg10s = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            cand = batch['candidate_title'].to(device)
            clkd = batch['clicked_title'].to(device)
            lbl = batch['clicked'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            logits, _ = model(cand, clkd)  # (1, num_candidates)
            
            # Bỏ padding
            mask_bool = mask.bool().squeeze(0)
            logits_valid = logits.squeeze(0)[mask_bool].cpu().numpy()
            labels_valid = lbl.squeeze(0)[mask_bool].cpu().numpy()
            
            # Tính metrics
            if len(np.unique(labels_valid)) > 1:  # Cần cả positive và negative
                auc, mrr, ndcg5, ndcg10 = calculate_single_user_metrics(labels_valid, logits_valid)
                all_aucs.append(auc)
                all_mrrs.append(mrr)
                all_ndcg5s.append(ndcg5)
                all_ndcg10s.append(ndcg10)
            else:
                # Nếu chỉ có 1 class (rất hiếm), bỏ qua
                pass
    
    model.train()
    
    return (
        np.nanmean(all_aucs),
        np.nanmean(all_mrrs),
        np.nanmean(all_ndcg5s),
        np.nanmean(all_ndcg10s)
    )

# ---------- 14. Huấn luyện ----------
num_epochs = 5
topic_loss_weight = 0.2
best_auc = 0.0

print("\n" + "="*70)
print("BẮT ĐẦU HUẤN LUYỆN TANR")
print("="*70)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        cand = batch['candidate_title'].to(device)
        clkd = batch['clicked_title'].to(device)
        lbl = batch['clicked'].to(device)
        
        optimizer.zero_grad()
        logits, topic_pred = model(cand, clkd)
        
        # Click loss
        target = torch.zeros(logits.size(0), dtype=torch.long).to(device)
        loss_click = criterion(logits, target)
        
        # Topic loss
        loss_topic = topic_criterion(topic_pred, target)
        
        # Combined
        loss = loss_click + topic_loss_weight * loss_topic
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        pred_class = logits.argmax(dim=-1)
        correct_preds += (pred_class == target).sum().item()
        total_preds += logits.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_preds/max(1,total_preds):.3f}'
        })
    
    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct_preds / max(1, total_preds)
    
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # Validation
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = validate(model, val_loader, device)
    
    print(f"  Val AUC:    {val_auc:.4f}")
    print(f"  Val MRR:    {val_mrr:.4f}")
    print(f"  Val nDCG@5: {val_ndcg5:.4f}")
    print(f"  Val nDCG@10: {val_ndcg10:.4f}")
    
    # Lưu model tốt nhất
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': {
                'auc': val_auc, 'mrr': val_mrr, 
                'ndcg5': val_ndcg5, 'ndcg10': val_ndcg10
            }
        }, 'tanr_best_model.pth')
        print(f"  ✓ Model tốt nhất đã lưu! (AUC: {best_auc:.4f})")
    
    print("-"*70)

print("\n" + "="*70)
print("HUẤN LUYỆN HOÀN TẤT!")
print("="*70)

# So sánh với paper
print("\n📊 KẾT QUẢ SO SÁNH VỚI PAPER:")
print("-" * 50)
print(f"{'Metric':<15} {'Paper TANR':<15} {'Ours TANR':<15}")
print("-" * 50)
print(f"{'AUC':<15} {'0.6338':<15} {val_auc:<15.4f}")
print(f"{'MRR':<15} {'0.2868':<15} {val_mrr:<15.4f}")
print(f"{'nDCG@5':<15} {'0.3169':<15} {val_ndcg5:<15.4f}")
print(f"{'nDCG@10':<15} {'0.3804':<15} {val_ndcg10:<15.4f}")
print("-" * 50)