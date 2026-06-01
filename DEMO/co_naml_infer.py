"""
Co-NAML-LSTUR inference service for Flask.

This file matches the trained checkpoint:
    DEMO/best_co_naml_lstur.pt

Input news fields:
    title, abstract, category, subcategory

User signal:
    MIND user_id -> user embedding
    clicked/read news history -> LSTUR user encoder
"""
from __future__ import annotations

import os
import random
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


DEFAULT_CONFIG: dict = {
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
    "dataset_attributes": {
        "news": ["category", "subcategory", "title", "abstract"]
    },
}


def _load_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pack_padded_sequence

        return torch, nn, F, pack_padded_sequence

    except OSError as e:
        raise OSError(
            "Cannot load PyTorch DLL. "
            "Please reinstall CPU PyTorch or repair Microsoft Visual C++ Redistributable. "
            f"Original error: {e}"
        )

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


def _read_behaviors(path: Path) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    return pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str)


def _load_news_and_users(root: Path) -> tuple[pd.DataFrame, dict, dict[str, int]]:
    train_news_path = root / "MINDsmall_train" / "news.tsv"
    val_news_path = root / "MINDsmall_val" / "news.tsv"
    train_behavior_path = root / "MINDsmall_train" / "behaviors.tsv"
    val_behavior_path = root / "MINDsmall_val" / "behaviors.tsv"

    for path in [
        train_news_path,
        val_news_path,
        train_behavior_path,
        val_behavior_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing MIND file: {path}")

    keep = ["news_id", "category", "subcategory", "title", "abstract"]

    train_news = _read_news(train_news_path)[keep]
    val_news = _read_news(val_news_path)[keep]

    all_news = pd.concat([train_news, val_news], ignore_index=True).fillna("")
    all_news = all_news.drop_duplicates(subset=["news_id"], keep="first").reset_index(drop=True)

    cat_encoder = LabelEncoder()
    subcat_encoder = LabelEncoder()

    cat_encoder.fit(all_news["category"].unique())
    subcat_encoder.fit(all_news["subcategory"].unique())

    all_news["cat_id"] = cat_encoder.transform(all_news["category"]) + 1
    all_news["subcat_id"] = subcat_encoder.transform(all_news["subcategory"]) + 1

    behaviors_train = _read_behaviors(train_behavior_path)
    behaviors_val = _read_behaviors(val_behavior_path)

    users = set(behaviors_train["user_id"].dropna().unique()) | set(
        behaviors_val["user_id"].dropna().unique()
    )

    user2idx = {u: i + 1 for i, u in enumerate(sorted(users))}

    config = dict(DEFAULT_CONFIG)
    config["num_categories"] = int(
        max(all_news["cat_id"].max(), all_news["subcat_id"].max()) + 1
    )
    config["num_users"] = int(len(user2idx) + 1)

    return all_news, config, user2idx


def build_model_classes():
    torch, nn, F, pack_padded_sequence = _load_torch()

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

    class DistilBertTextEncoder(nn.Module):
        def __init__(self, num_filters, query_vector_dim, dropout_probability, max_length=64):
            super().__init__()

            from transformers import DistilBertModel, DistilBertTokenizer

            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert = DistilBertModel.from_pretrained(
                "distilbert-base-uncased"
            )

            for param in self.bert.parameters():
                param.requires_grad = False

            self.projection = nn.Sequential(
                nn.Linear(768, num_filters * 2),
                nn.ReLU(),
                nn.Dropout(dropout_probability),
                nn.Linear(num_filters * 2, num_filters),
            )

            self.attention = AdditiveAttention(query_vector_dim, num_filters)
            self.max_length = max_length

        def forward(self, texts):
            if isinstance(texts, str):
                texts = [texts]

            encoded = self.tokenizer(
                texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            device = next(self.parameters()).device

            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                )

            embeddings = outputs.last_hidden_state
            projected = self.projection(embeddings)
            return self.attention(projected)

    class ElementEncoder(nn.Module):
        def __init__(self, num_categories, embedding_dim, output_dim):
            super().__init__()

            self.embedding = nn.Embedding(
                num_categories,
                embedding_dim,
                padding_idx=0,
            )
            self.linear = nn.Linear(embedding_dim, output_dim)

            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.linear.weight)

        def forward(self, element):
            return F.relu(self.linear(self.embedding(element)))

    class NewsEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.text_encoders = nn.ModuleDict(
                {
                    "title": DistilBertTextEncoder(
                        config["num_filters"],
                        config["query_vector_dim"],
                        config["dropout_probability"],
                        max_length=config["max_title_len"],
                    ),
                    "abstract": DistilBertTextEncoder(
                        config["num_filters"],
                        config["query_vector_dim"],
                        config["dropout_probability"],
                        max_length=config["max_abstract_len"],
                    ),
                }
            )

            self.element_encoders = nn.ModuleDict(
                {
                    "category": ElementEncoder(
                        config["num_categories"],
                        config["category_embedding_dim"],
                        config["num_filters"],
                    ),
                    "subcategory": ElementEncoder(
                        config["num_categories"],
                        config["category_embedding_dim"],
                        config["num_filters"],
                    ),
                }
            )

            self.final_attention = AdditiveAttention(
                config["query_vector_dim"],
                config["num_filters"],
            )

        def forward(self, news):
            vectors = []

            if "title" in news and news["title"] is not None:
                vectors.append(self.text_encoders["title"](news["title"]))

            if "abstract" in news and news["abstract"] is not None:
                vectors.append(self.text_encoders["abstract"](news["abstract"]))

            if "category" in news and news["category"] is not None:
                vectors.append(
                    self.element_encoders["category"](
                        news["category"].to(next(self.parameters()).device)
                    )
                )

            if "subcategory" in news and news["subcategory"] is not None:
                vectors.append(
                    self.element_encoders["subcategory"](
                        news["subcategory"].to(next(self.parameters()).device)
                    )
                )

            if len(vectors) == 1:
                return vectors[0]

            return self.final_attention(torch.stack(vectors, dim=1))

    class DKNAttention(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.dnn = nn.Sequential(
                nn.Linear(config["num_filters"] * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, candidate_news_vector, clicked_news_vector):
            num_clicked = clicked_news_vector.size(1)

            expanded_candidate = candidate_news_vector.unsqueeze(1).expand(
                -1,
                num_clicked,
                -1,
            )

            concat_vectors = torch.cat(
                (expanded_candidate, clicked_news_vector),
                dim=2,
            )

            attention_scores = self.dnn(concat_vectors).squeeze(dim=2)
            clicked_news_weights = F.softmax(attention_scores, dim=1)

            return torch.bmm(
                clicked_news_weights.unsqueeze(1),
                clicked_news_vector,
            ).squeeze(1)

    class UserEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()

            # Checkpoint:
            # lstm.weight_ih_l0 shape [1800, 300]
            # => 4 * hidden_size = 1800
            # => hidden_size = 450
            self.hidden_size = int(config["num_filters"] * 1.5)

            # Checkpoint cần output cuối cho linear là 900.
            # Vì vậy dùng bidirectional=True:
            # 450 forward + 450 backward = 900.
            self.lstm = nn.LSTM(
                config["num_filters"],
                self.hidden_size,
                batch_first=True,
                bidirectional=True,
            )

            self.linear = nn.Linear(
                self.hidden_size * 2,
                config["num_filters"],
            )

        def forward(self, long_term_emb, clicked_news_length, clicked_news_vector):
            lengths = clicked_news_length.clone()
            lengths[lengths == 0] = 1

            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            sorted_vectors = clicked_news_vector[sorted_idx]
            sorted_long_term = long_term_emb[sorted_idx]

            packed = pack_padded_sequence(
                sorted_vectors,
                sorted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )

            h0 = sorted_long_term.unsqueeze(0).repeat(2, 1, 1)
            c0 = torch.zeros_like(h0)

            _, (last_hidden, _) = self.lstm(packed, (h0, c0))

            u_s_sorted = torch.cat(
                [last_hidden[0], last_hidden[1]],
                dim=1,
            )

            _, unsorted_idx = torch.sort(sorted_idx)

            return self.linear(u_s_sorted[unsorted_idx])

    class DNNClickPredictor(nn.Module):
        def __init__(self, input_size, hidden_size=128):
            super().__init__()

            # Checkpoint:
            # click_predictor.dnn.0.weight shape [128, 600]
            self.dnn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, candidate_news_vector, user_vector):
            return self.dnn(
                torch.cat(
                    (candidate_news_vector, user_vector),
                    dim=1,
                )
            ).squeeze(1)

    class Co_NAML_LSTUR(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.config = config
            self.news_encoder = NewsEncoder(config)
            self.user_encoder = UserEncoder(config)

            # Checkpoint:
            # user_embedding.weight shape [94058, 450]
            self.user_embedding = nn.Embedding(
                config["num_users"],
                int(config["num_filters"] * 1.5),
                padding_idx=0,
            )

            nn.init.xavier_uniform_(self.user_embedding.weight)

            self.attention = DKNAttention(config)
            self.click_predictor = DNNClickPredictor(
                input_size=config["num_filters"] * 2,
                hidden_size=128,
            )

        def forward(
            self,
            user,
            clicked_news_length,
            candidate_news_vectors,
            clicked_news_vectors,
        ):
            B, N, D = candidate_news_vectors.shape
            S = clicked_news_vectors.shape[1]

            u_l = F.dropout(
                self.user_embedding(user),
                p=self.config["masking_probability"],
                training=self.training,
            )

            u_s = self.user_encoder(
                u_l,
                clicked_news_length,
                clicked_news_vectors,
            )

            candidate_flat = candidate_news_vectors.reshape(B * N, D)

            clicked_expanded = (
                clicked_news_vectors.unsqueeze(1)
                .expand(-1, N, -1, -1)
                .reshape(B * N, S, D)
            )

            u_att = self.attention(
                candidate_flat,
                clicked_expanded,
            ).view(B, N, D)

            user_vector = u_s.unsqueeze(1) * u_att

            scores = self.click_predictor(
                candidate_news_vectors.reshape(B * N, D),
                user_vector.reshape(B * N, D),
            ).view(B, N)

            return scores

    return torch, Co_NAML_LSTUR


class CoNAMLService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

        self._all_news: pd.DataFrame | None = None
        self._news_index: dict[str, int] = {}
        self._user2idx: dict[str, int] = {}
        self._config: dict | None = None

        self._model = None
        self._torch = None
        self._device = None

        self._load_error: str | None = None
        self._checkpoint_path = self.root / "best_co_naml_lstur.pt"

        self._checkpoint_loaded = False
        self._ckpt_missing_keys = 0
        self._ckpt_unexpected_keys = 0

        self._news_vector_cache: dict[str, object] = {}

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None and self._all_news is not None

    @property
    def load_error(self) -> str | None:
        self._ensure()
        return self._load_error

    def _safe_torch_load(self, torch):
        try:
            return torch.load(
                self._checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                self._checkpoint_path,
                map_location=self._device,
            )

    def _extract_state_dict(self, loaded_obj):
        if isinstance(loaded_obj, dict):
            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
            ]:
                if key in loaded_obj and isinstance(loaded_obj[key], dict):
                    return loaded_obj[key]

            return loaded_obj

        raise RuntimeError(
            "Checkpoint is not a state_dict. "
            f"Loaded type: {type(loaded_obj)}"
        )

    def _ensure(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            if os.environ.get("DISABLE_CONAML", "").lower() in ("1", "true", "yes"):
                self._load_error = "DISABLE_CONAML is set."
                return

            try:
                if not self._checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Missing checkpoint: {self._checkpoint_path}"
                    )

                torch, Co_NAML_LSTUR = build_model_classes()

                self._torch = torch
                self._device = torch.device("cpu")

                all_news, config, user2idx = _load_news_and_users(self.root)

                model = Co_NAML_LSTUR(config).to(self._device)
                model.eval()

                loaded = self._safe_torch_load(torch)
                state_dict = self._extract_state_dict(loaded)

                missing, unexpected = model.load_state_dict(
                    state_dict,
                    strict=False,
                )

                self._ckpt_missing_keys = len(missing)
                self._ckpt_unexpected_keys = len(unexpected)
                self._checkpoint_loaded = True

                self._all_news = all_news
                self._news_index = {
                    nid: i
                    for i, nid in enumerate(all_news["news_id"].tolist())
                }
                self._user2idx = user2idx
                self._config = config
                self._model = model
                self._load_error = None

            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                self._all_news = None
                self._news_index = {}
                self._user2idx = {}
                self._config = None
                self._model = None

    def status_message(self) -> str:
        self._ensure()

        if self._load_error:
            return f"Co-NAML-LSTUR unavailable: {self._load_error}"

        if not self.available:
            return "Co-NAML-LSTUR unavailable."

        return (
            f"Co-NAML-LSTUR loaded from `{self._checkpoint_path.name}`. "
            f"Missing keys: {self._ckpt_missing_keys}, "
            f"unexpected keys: {self._ckpt_unexpected_keys}."
        )

    def _encode_news_rows(self, rows_df: pd.DataFrame):
        torch = self._torch
        model = self._model

        news_batch = {
            "title": rows_df["title"].tolist(),
            "abstract": rows_df["abstract"].tolist(),
            "category": torch.tensor(
                rows_df["cat_id"].values,
                dtype=torch.long,
                device=self._device,
            ),
            "subcategory": torch.tensor(
                rows_df["subcat_id"].values,
                dtype=torch.long,
                device=self._device,
            ),
        }

        return model.news_encoder(news_batch)

    def _get_news_vectors(self, news_ids: list[str]):
        torch = self._torch

        missing_ids = [
            nid
            for nid in news_ids
            if nid not in self._news_vector_cache
        ]

        if missing_ids:
            rows = []
            valid_missing_ids = []

            for nid in missing_ids:
                idx = self._news_index.get(nid)
                if idx is None:
                    continue

                rows.append(self._all_news.iloc[idx])
                valid_missing_ids.append(nid)

            if rows:
                rows_df = pd.DataFrame(rows)

                with torch.no_grad():
                    vecs = self._encode_news_rows(rows_df).detach().cpu()

                for nid, vec in zip(valid_missing_ids, vecs):
                    self._news_vector_cache[nid] = vec

        vectors = [
            self._news_vector_cache[nid]
            for nid in news_ids
            if nid in self._news_vector_cache
        ]

        if not vectors:
            return None

        return torch.stack(vectors).to(self._device)

    def _candidate_pool(self, history_ids: list[str], max_candidates: int) -> list[str]:
        exclude = set(history_ids)

        pool = [
            nid
            for nid in self._all_news["news_id"].tolist()
            if nid not in exclude
        ]

        if len(pool) > max_candidates:
            random.seed(42)
            pool = random.sample(pool, max_candidates)

        return pool

    def recommend_trained(
        self,
        history_news_ids: list[str],
        k: int = 20,
        max_history: int = 50,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self.recommend(
            history_news_ids=history_news_ids,
            candidate_news_ids=None,
            k=k,
            max_history=max_history,
            max_candidates=100,
            user_id=user_id,
        )

    def recommend(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str] | None = None,
        k: int = 16,
        max_history: int = 50,
        max_candidates: int = 100,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        self._ensure()

        if not self.available or self._model is None or self._all_news is None:
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

        if candidate_news_ids:
            cids = [
                nid
                for nid in candidate_news_ids
                if nid in valid_ids and nid not in set(hist_ids)
            ]
        else:
            cids = self._candidate_pool(hist_ids, max_candidates)

        cids = cids[:max_candidates]

        if not cids:
            return []

        clicked_vectors = self._get_news_vectors(hist_ids)
        candidate_vectors = self._get_news_vectors(cids)

        if clicked_vectors is None or candidate_vectors is None:
            return []

        clicked_vectors = clicked_vectors.unsqueeze(0)
        candidate_vectors = candidate_vectors.unsqueeze(0)

        clicked_len = torch.tensor(
            [clicked_vectors.size(1)],
            dtype=torch.long,
            device=self._device,
        )

        user_idx = 0

        if user_id:
            user_idx = int(self._user2idx.get(user_id, 0))

        user_tensor = torch.tensor(
            [user_idx],
            dtype=torch.long,
            device=self._device,
        )

        self._model.eval()

        with torch.no_grad():
            scores = self._model(
                user_tensor,
                clicked_len,
                candidate_vectors,
                clicked_vectors,
            )
            probs = torch.sigmoid(scores).squeeze(0).cpu().numpy()

        order = np.argsort(-probs)[:k]

        results: list[tuple[str, float]] = []

        for idx in order:
            results.append(
                (
                    str(cids[int(idx)]),
                    float(probs[int(idx)]),
                )
            )

        return results


_svc: CoNAMLService | None = None
_svc_lock = threading.Lock()


def get_co_naml_service(root: Path | None = None) -> CoNAMLService:
    global _svc

    with _svc_lock:
        if _svc is None:
            r = root or Path(__file__).resolve().parent
            _svc = CoNAMLService(r)

        return _svc