from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _load_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as e:
        raise RuntimeError(f"Cannot import PyTorch: {type(e).__name__}: {e}")


def _read_news(path: Path) -> pd.DataFrame:
    cols = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    return pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str)


def _load_news_table(root: Path) -> pd.DataFrame:
    train_path = root / "MINDsmall_train" / "news.tsv"
    val_path = root / "MINDsmall_val" / "news.tsv"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing file: {val_path}")

    keep = ["news_id", "category", "subcategory", "title", "abstract"]
    train_news = _read_news(train_path)[keep]
    val_news = _read_news(val_path)[keep]

    df = pd.concat([train_news, val_news], ignore_index=True).fillna("")
    df = df.drop_duplicates(subset=["news_id"], keep="first").reset_index(drop=True)
    return df


def build_nrms_model():
    torch, nn, F = _load_torch()

    class AdditiveAttention(nn.Module):
        def __init__(self, input_dim: int, query_dim: int = 200):
            super().__init__()
            self.linear = nn.Linear(input_dim, query_dim)
            self.query = nn.Parameter(torch.randn(query_dim))

        def forward(self, x):
            a = torch.tanh(self.linear(x))
            weights = torch.softmax(torch.matmul(a, self.query), dim=1)
            return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

    class DistilBertNewsEncoder(nn.Module):
        def __init__(self, output_dim: int = 256, max_len: int = 30):
            super().__init__()
            from transformers import DistilBertModel, DistilBertTokenizer

            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

            for p in self.bert.parameters():
                p.requires_grad = False

            self.proj = nn.Linear(768, output_dim)
            self.attention = AdditiveAttention(output_dim)
            self.max_len = max_len

        def forward(self, titles: list[str]):
            device = next(self.parameters()).device
            encoded = self.tokenizer(
                titles,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            token_vecs = self.proj(out.last_hidden_state)
            news_vec = self.attention(token_vecs)
            return news_vec

    class NRMS(nn.Module):
        def __init__(self, news_dim: int = 256):
            super().__init__()
            self.news_encoder = DistilBertNewsEncoder(output_dim=news_dim)
            self.user_attention = AdditiveAttention(news_dim)

        def encode_news(self, titles: list[str]):
            return self.news_encoder(titles)

        def encode_user(self, clicked_news_vectors):
            if clicked_news_vectors.dim() == 2:
                clicked_news_vectors = clicked_news_vectors.unsqueeze(0)
            return self.user_attention(clicked_news_vectors)

        def forward(self, clicked_news_vectors, candidate_news_vectors):
            import torch.nn.functional as F
            # Normalize embeddings
            clicked_news_vectors = F.normalize(clicked_news_vectors, p=2, dim=-1)
            candidate_news_vectors = F.normalize(candidate_news_vectors, p=2, dim=-1)

            user_vec = self.encode_user(clicked_news_vectors)
            if candidate_news_vectors.dim() == 2:
                candidate_news_vectors = candidate_news_vectors.unsqueeze(0)

            # Cosine similarity -> [-1,1], then map to [0,1] click probability
            scores = torch.bmm(candidate_news_vectors, user_vec.unsqueeze(-1)).view(-1)
            probs = ((scores + 1.0) / 2.0).cpu().numpy()
            return probs

    return torch, NRMS


class NRMSService:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._checkpoint_path = root / "best_nrms_distilbert.pth"

        self._torch = None
        self._device = None
        self._model = None
        self._news_df: pd.DataFrame | None = None
        self._news_index: dict[str, int] = {}
        self._vector_cache: dict[str, object] = {}

        self._load_error: str | None = None
        self._missing_keys = 0
        self._unexpected_keys = 0

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None and self._news_df is not None

    @property
    def load_error(self) -> str | None:
        self._ensure()
        return self._load_error

    def _safe_load(self, torch):
        try:
            return torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
        except TypeError:
            return torch.load(self._checkpoint_path, map_location=self._device)

    def _extract_state_dict(self, obj):
        if isinstance(obj, dict):
            for key in ["state_dict", "model_state_dict", "model", "net"]:
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
            return obj
        raise RuntimeError(f"Checkpoint type is not supported: {type(obj)}")

    def _ensure(self):
        with self._lock:
            if self._model is not None:
                return

            try:
                if not self._checkpoint_path.is_file():
                    raise FileNotFoundError(f"Missing checkpoint: {self._checkpoint_path}")

                torch, NRMS = build_nrms_model()
                self._torch = torch
                self._device = torch.device("cpu")

                torch.manual_seed(42)
                np.random.seed(42)

                model = NRMS(news_dim=256).to(self._device)
                model.eval()

                loaded = self._safe_load(torch)
                state_dict = self._extract_state_dict(loaded)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                self._missing_keys = len(missing)
                self._unexpected_keys = len(unexpected)

                news_df = _load_news_table(self.root)
                self._news_df = news_df
                self._news_index = {str(nid): i for i, nid in enumerate(news_df["news_id"].tolist())}
                self._model = model
                self._load_error = None

            except Exception as e:
                self._model = None
                self._news_df = None
                self._news_index = {}
                self._load_error = f"{type(e).__name__}: {e}"

    def status_message(self) -> str:
        self._ensure()
        if self._load_error:
            return f"NRMS DistilBERT unavailable: {self._load_error}"
        return (
            f"NRMS DistilBERT loaded from `{self._checkpoint_path.name}`. "
            f"Missing keys: {self._missing_keys}, unexpected keys: {self._unexpected_keys}."
        )

    def _encode_news_ids(self, news_ids: list[str]):
        torch = self._torch
        missing = [nid for nid in news_ids if nid not in self._vector_cache and nid in self._news_index]
        if missing:
            rows = [self._news_df.iloc[self._news_index[nid]] for nid in missing]
            titles = [str(row["title"]) for row in rows]
            with torch.no_grad():
                vecs = self._model.encode_news(titles).detach().cpu()
            for nid, vec in zip(missing, vecs):
                self._vector_cache[nid] = vec

        vectors = [self._vector_cache[nid] for nid in news_ids if nid in self._vector_cache]
        if not vectors:
            return None

        return torch.stack(vectors).to(self._device)

    def recommend_trained(
        self,
        history_news_ids: list[str],
        k: int = 12,
        max_history: int = 50,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self.recommend(
            history_news_ids=history_news_ids,
            candidate_news_ids=None,
            k=k,
            max_history=max_history,
            max_candidates=100,
        )

    def recommend(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str] | None = None,
        k: int = 12,
        max_history: int = 50,
        max_candidates: int = 100,
    ) -> list[tuple[str, float]]:
        self._ensure()
        if not self.available or self._model is None or self._news_df is None:
            return []

        torch = self._torch
        valid_ids = set(self._news_index.keys())
        hist_ids: list[str] = []
        seen: set[str] = set()
        for nid in history_news_ids:
            if nid in valid_ids and nid not in seen:
                seen.add(nid)
                hist_ids.append(nid)
        hist_ids = hist_ids[-max_history:]
        if not hist_ids:
            return []

        exclude = set(hist_ids)
        if candidate_news_ids:
            cids = [nid for nid in candidate_news_ids if nid in valid_ids and nid not in exclude]
        else:
            pool = [str(nid) for nid in self._news_df["news_id"].tolist() if str(nid) not in exclude]
            cids = pool[:max_candidates]
        if not cids:
            return []

        h_vecs = self._encode_news_ids(hist_ids)
        c_vecs = self._encode_news_ids(cids)
        if h_vecs is None or c_vecs is None:
            return []

        self._model.eval()
        with torch.no_grad():
            import torch.nn.functional as F
            # Normalize embeddings
            h_vecs = F.normalize(h_vecs, p=2, dim=-1)
            c_vecs = F.normalize(c_vecs, p=2, dim=-1)

            if h_vecs.dim() == 2:
                h_vecs = h_vecs.unsqueeze(0)
            if c_vecs.dim() == 2:
                c_vecs = c_vecs.unsqueeze(0)

            user_vec = self._model.encode_user(h_vecs)
            scores = torch.bmm(c_vecs, user_vec.unsqueeze(-1)).view(-1)
            probs = ((scores + 1.0) / 2.0).cpu().numpy()  # Map [-1,1] -> [0,1]

        order = np.argsort(-probs)[:k]
        return [(str(cids[int(i)]), float(probs[int(i)])) for i in order]


_svc: NRMSService | None = None
_svc_lock = threading.Lock()


def get_nrms_service(root: Path | None = None) -> NRMSService:
    global _svc
    with _svc_lock:
        if _svc is None:
            r = root or Path(__file__).resolve().parent
            _svc = NRMSService(r)
        return _svc