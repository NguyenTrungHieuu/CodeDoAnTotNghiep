from __future__ import annotations

import threading
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


HISTORY_LEN = 30
MAX_TITLE_LEN = 30
MAX_ABS_LEN = 50

NUM_FILTERS = 256
QUERY_VECTOR_DIM = 128
DROPOUT_RATE = 0.3

# Giảm candidate để NAML chạy nhanh hơn trong demo Flask.
# Model vẫn xếp hạng top-k theo predicted click score.
DEFAULT_MAX_CANDIDATES = 60
ENCODE_BATCH_SIZE = 16


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

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=cols,
        dtype=str,
    )


def _read_behaviors(path: Path) -> pd.DataFrame:
    cols = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=cols,
        dtype=str,
    )


def _parse_impressions(cell: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []

    if pd.isna(cell) or not str(cell).strip():
        return out

    for part in str(cell).split():
        if "-" not in part:
            continue

        news_id, label = part.rsplit("-", 1)

        try:
            out.append((news_id, int(label)))
        except ValueError:
            continue

    return out


def _load_news_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = root / "MINDsmall_train" / "news.tsv"
    val_path = root / "MINDsmall_val" / "news.tsv"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not val_path.is_file():
        raise FileNotFoundError(f"Missing file: {val_path}")

    train_news = _read_news(train_path).fillna("")
    val_news = _read_news(val_path).fillna("")

    all_news = pd.concat(
        [train_news, val_news],
        ignore_index=True,
    ).fillna("")

    all_news = all_news.drop_duplicates(
        subset=["news_id"],
        keep="first",
    ).reset_index(drop=True)

    all_news["title"] = all_news["title"].fillna("")
    all_news["abstract"] = all_news["abstract"].fillna("")
    all_news["category"] = all_news["category"].fillna("unknown")
    all_news["subcategory"] = all_news["subcategory"].fillna("unknown")

    return train_news, val_news, all_news


def _load_behaviors(root: Path) -> pd.DataFrame:
    train_path = root / "MINDsmall_train" / "behaviors.tsv"
    val_path = root / "MINDsmall_val" / "behaviors.tsv"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not val_path.is_file():
        raise FileNotFoundError(f"Missing file: {val_path}")

    train_behaviors = _read_behaviors(train_path)
    val_behaviors = _read_behaviors(val_path)

    return pd.concat(
        [train_behaviors, val_behaviors],
        ignore_index=True,
    ).fillna("")


def build_naml_model():
    torch, nn, F = _load_torch()

    class AdditiveAttention(nn.Module):
        def __init__(
            self,
            query_vector_dim: int,
            candidate_vector_dim: int,
        ):
            super().__init__()

            self.linear = nn.Linear(
                candidate_vector_dim,
                query_vector_dim,
            )

            self.query_vector = nn.Parameter(
                torch.randn(query_vector_dim, 1)
            )

        def forward(self, candidate_vector):
            projected = torch.tanh(self.linear(candidate_vector))

            attention_scores = torch.matmul(
                projected,
                self.query_vector,
            ).squeeze(-1)

            attention_weights = F.softmax(
                attention_scores,
                dim=1,
            )

            weighted_sum = torch.bmm(
                candidate_vector.transpose(1, 2),
                attention_weights.unsqueeze(-1),
            ).squeeze(-1)

            return weighted_sum

    class TextEncoder(nn.Module):
        def __init__(
            self,
            distilbert,
            hidden_size: int,
            num_filters: int,
            query_vector_dim: int,
            dropout_probability: float,
        ):
            super().__init__()

            self.distilbert = distilbert
            self.dropout_probability = dropout_probability

            self.projection = nn.Linear(
                hidden_size,
                num_filters,
            )

            self.additive_attention = AdditiveAttention(
                query_vector_dim,
                num_filters,
            )

        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                bert_output = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                token_emb = bert_output.last_hidden_state

            projected = self.projection(token_emb)

            projected = F.dropout(
                projected,
                p=self.dropout_probability,
                training=self.training,
            )

            return self.additive_attention(projected)

    class ElementEncoder(nn.Module):
        def __init__(
            self,
            embedding,
            linear_input_dim: int,
            linear_output_dim: int,
        ):
            super().__init__()

            self.embedding = embedding

            self.linear = nn.Linear(
                linear_input_dim,
                linear_output_dim,
            )

        def forward(self, element):
            element = torch.clamp(
                element,
                0,
                self.embedding.num_embeddings - 1,
            )

            return F.relu(
                self.linear(
                    self.embedding(element)
                )
            )

    class NewsEncoder(nn.Module):
        def __init__(
            self,
            distilbert,
            hidden_size: int,
            num_categories: int,
            num_subcategories: int,
        ):
            super().__init__()

            self.text_encoders = nn.ModuleDict(
                {
                    "title": TextEncoder(
                        distilbert,
                        hidden_size,
                        NUM_FILTERS,
                        QUERY_VECTOR_DIM,
                        DROPOUT_RATE,
                    ),
                    "abstract": TextEncoder(
                        distilbert,
                        hidden_size,
                        NUM_FILTERS,
                        QUERY_VECTOR_DIM,
                        DROPOUT_RATE,
                    ),
                }
            )

            self.cat_embedding = nn.Embedding(
                num_categories,
                NUM_FILTERS,
                padding_idx=0,
            )

            self.subcat_embedding = nn.Embedding(
                num_subcategories,
                NUM_FILTERS,
                padding_idx=0,
            )

            self.element_encoders = nn.ModuleDict(
                {
                    "category": ElementEncoder(
                        self.cat_embedding,
                        NUM_FILTERS,
                        NUM_FILTERS,
                    ),
                    "subcategory": ElementEncoder(
                        self.subcat_embedding,
                        NUM_FILTERS,
                        NUM_FILTERS,
                    ),
                }
            )

            self.final_attention = AdditiveAttention(
                QUERY_VECTOR_DIM,
                NUM_FILTERS,
            )

        def forward(self, news):
            batch_size = news["title"].size(0)
            num_news = news["title"].size(1)

            title_flat = news["title"].view(
                batch_size * num_news,
                -1,
            )

            title_mask_flat = news["title_mask"].view(
                batch_size * num_news,
                -1,
            )

            abs_flat = news["abstract"].view(
                batch_size * num_news,
                -1,
            )

            abs_mask_flat = news["abstract_mask"].view(
                batch_size * num_news,
                -1,
            )

            cat_flat = news["category"].view(
                batch_size * num_news
            )

            subcat_flat = news["subcategory"].view(
                batch_size * num_news
            )

            title_vec = self.text_encoders["title"](
                title_flat,
                title_mask_flat,
            )

            abs_vec = self.text_encoders["abstract"](
                abs_flat,
                abs_mask_flat,
            )

            cat_vec = self.element_encoders["category"](
                cat_flat
            )

            subcat_vec = self.element_encoders["subcategory"](
                subcat_flat
            )

            all_vectors = torch.stack(
                [
                    title_vec,
                    abs_vec,
                    cat_vec,
                    subcat_vec,
                ],
                dim=1,
            )

            final_flat = self.final_attention(all_vectors)

            return final_flat.view(
                batch_size,
                num_news,
                -1,
            )

    class UserEncoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.additive_attention = AdditiveAttention(
                QUERY_VECTOR_DIM,
                NUM_FILTERS,
            )

        def forward(self, clicked_news_vector):
            return self.additive_attention(clicked_news_vector)

    class NAML(nn.Module):
        def __init__(
            self,
            num_categories: int,
            num_subcategories: int,
            distilbert,
            hidden_size: int,
        ):
            super().__init__()

            self.news_encoder = NewsEncoder(
                distilbert,
                hidden_size,
                num_categories,
                num_subcategories,
            )

            self.user_encoder = UserEncoder()

        def forward(
            self,
            h_title,
            h_title_mask,
            h_abs,
            h_abs_mask,
            h_cat,
            h_sub,
            c_title,
            c_title_mask,
            c_abs,
            c_abs_mask,
            c_cat,
            c_sub,
        ):
            h_news = {
                "title": h_title,
                "title_mask": h_title_mask,
                "abstract": h_abs,
                "abstract_mask": h_abs_mask,
                "category": h_cat,
                "subcategory": h_sub,
            }

            h_vecs = self.news_encoder(h_news)
            user_vec = self.user_encoder(h_vecs)

            c_news = {
                "title": c_title,
                "title_mask": c_title_mask,
                "abstract": c_abs,
                "abstract_mask": c_abs_mask,
                "category": c_cat,
                "subcategory": c_sub,
            }

            c_vecs = self.news_encoder(c_news)

            scores = torch.bmm(
                c_vecs,
                user_vec.unsqueeze(-1),
            ).squeeze(-1)

            return scores

    return torch, NAML


class NAMLService:
    def __init__(
        self,
        root: Path,
        checkpoint: str = "naml_distilbert_best.pt",
    ):
        self.root = root
        self._checkpoint_path = root / checkpoint
        self._lock = threading.Lock()

        self._torch = None
        self._device = None
        self._tokenizer = None
        self._model = None

        self._news_df: pd.DataFrame | None = None
        self._behaviors: pd.DataFrame | None = None

        self._news_index: dict[str, int] = {}
        self._cat_dict: dict[str, int] = {}
        self._subcat_dict: dict[str, int] = {}

        self._news_features: dict[str, dict] = {}

        self._news_vector_cache: dict[str, object] = {}

        self._user_impression_candidates: dict[str, list[str]] = defaultdict(list)
        self._click_counts: Counter[str] = Counter()

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

    def _checkpoint_exists(self) -> Path:
        if self._checkpoint_path.is_file():
            return self._checkpoint_path

        candidates = [
            self.root / "naml_distilbert_best.pt",
            self.root / "naml_distilbert_best(5).pt",
            self.root / "naml_distilbert_best(4).pt",
            self.root / "naml_distilbert_best.pth",
            self.root / "NAML_distilbert_best.pt",
        ]

        for path in candidates:
            if path.is_file():
                self._checkpoint_path = path
                return path

        raise FileNotFoundError(
            f"Missing checkpoint: {self._checkpoint_path}"
        )

    def _safe_load(self, torch):
        ckpt = self._checkpoint_exists()

        try:
            return torch.load(
                ckpt,
                map_location=self._device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                ckpt,
                map_location=self._device,
            )

    def _extract_state_dict(self, obj):
        if isinstance(obj, dict):
            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "naml",
            ]:
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]

            return obj

        raise RuntimeError(
            f"Checkpoint type is not supported: {type(obj)}"
        )

    def _clean_state_dict_keys(self, state_dict: dict) -> dict:
        cleaned = {}

        for key, value in state_dict.items():
            new_key = str(key)

            for prefix in [
                "module.",
                "model.",
                "net.",
                "naml.",
            ]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            cleaned[new_key] = value

        return cleaned

    def _build_category_dicts(self, news_df: pd.DataFrame) -> None:
        cat_dict = {
            c: i + 1
            for i, c in enumerate(news_df["category"].unique())
        }

        cat_dict["unknown"] = 0

        subcat_dict = {
            c: i + 1
            for i, c in enumerate(news_df["subcategory"].unique())
        }

        subcat_dict["unknown"] = 0

        self._cat_dict = cat_dict
        self._subcat_dict = subcat_dict

    def _build_news_features(self, news_df: pd.DataFrame) -> None:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        titles = news_df["title"].fillna("").tolist()
        abstracts = news_df["abstract"].fillna("").tolist()

        title_enc = self._tokenizer(
            titles,
            max_length=MAX_TITLE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        abs_enc = self._tokenizer(
            abstracts,
            max_length=MAX_ABS_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        features: dict[str, dict] = {}

        for i, row in news_df.iterrows():
            news_id = str(row["news_id"])

            category = str(row.get("category") or "unknown")
            subcategory = str(row.get("subcategory") or "unknown")

            features[news_id] = {
                "title": title_enc["input_ids"][i].tolist(),
                "title_mask": title_enc["attention_mask"][i].tolist(),
                "abstract": abs_enc["input_ids"][i].tolist(),
                "abstract_mask": abs_enc["attention_mask"][i].tolist(),
                "category": int(self._cat_dict.get(category, 0)),
                "subcategory": int(self._subcat_dict.get(subcategory, 0)),
            }

        self._news_features = features

    def _build_user_candidate_cache(self) -> None:
        if self._behaviors is None:
            return

        user_candidates: dict[str, list[str]] = defaultdict(list)
        user_seen: dict[str, set[str]] = defaultdict(set)
        click_counts: Counter[str] = Counter()

        for _, row in self._behaviors.iterrows():
            user_id = str(row.get("user_id") or "").strip()

            if not user_id:
                continue

            for news_id, label in _parse_impressions(
                str(row.get("impressions") or "")
            ):
                if news_id not in self._news_index:
                    continue

                if news_id not in user_seen[user_id]:
                    user_seen[user_id].add(news_id)
                    user_candidates[user_id].append(news_id)

                if label == 1:
                    click_counts[news_id] += 1

        self._user_impression_candidates = user_candidates
        self._click_counts = click_counts

    def _ensure(self):
        with self._lock:
            if self._model is not None:
                return

            try:
                self._checkpoint_exists()

                torch, NAML = build_naml_model()

                from transformers import DistilBertModel, DistilBertTokenizer

                self._torch = torch
                self._device = torch.device("cpu")

                torch.manual_seed(42)
                np.random.seed(42)

                tokenizer = DistilBertTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )

                distilbert = DistilBertModel.from_pretrained(
                    "distilbert-base-uncased"
                ).to(self._device)

                for param in distilbert.parameters():
                    param.requires_grad = False

                distilbert.eval()

                hidden_size = distilbert.config.hidden_size

                _, _, news_df = _load_news_tables(self.root)
                behaviors = _load_behaviors(self.root)

                self._news_df = news_df
                self._behaviors = behaviors
                self._tokenizer = tokenizer

                self._news_index = {
                    str(news_id): i
                    for i, news_id in enumerate(news_df["news_id"].tolist())
                }

                self._build_category_dicts(news_df)
                self._build_news_features(news_df)

                model = NAML(
                    num_categories=len(self._cat_dict),
                    num_subcategories=len(self._subcat_dict),
                    distilbert=distilbert,
                    hidden_size=hidden_size,
                ).to(self._device)

                loaded = self._safe_load(torch)
                state_dict = self._extract_state_dict(loaded)
                state_dict = self._clean_state_dict_keys(state_dict)

                missing, unexpected = model.load_state_dict(
                    state_dict,
                    strict=False,
                )

                self._missing_keys = len(missing)
                self._unexpected_keys = len(unexpected)

                model.eval()

                self._model = model
                self._load_error = None

                self._build_user_candidate_cache()

            except Exception as e:
                self._model = None
                self._news_df = None
                self._behaviors = None
                self._news_index = {}
                self._news_features = {}
                self._news_vector_cache = {}
                self._load_error = f"{type(e).__name__}: {e}"

    def status_message(self) -> str:
        self._ensure()

        if self._load_error:
            return f"NAML unavailable: {self._load_error}"

        if self._missing_keys == 0 and self._unexpected_keys == 0:
            return (
                f"NAML loaded from `{self._checkpoint_path.name}`. "
                f"All checkpoint keys matched. "
                f"Output is predicted click score."
            )

        return (
            f"NAML loaded from `{self._checkpoint_path.name}`. "
            f"Missing keys: {self._missing_keys}, "
            f"unexpected keys: {self._unexpected_keys}. "
            f"Output is predicted click score."
        )

    def _news_tensor_batch(self, news_ids: list[str]):
        torch = self._torch

        pad = {
            "title": [0] * MAX_TITLE_LEN,
            "title_mask": [0] * MAX_TITLE_LEN,
            "abstract": [0] * MAX_ABS_LEN,
            "abstract_mask": [0] * MAX_ABS_LEN,
            "category": 0,
            "subcategory": 0,
        }

        titles = []
        title_masks = []
        abstracts = []
        abstract_masks = []
        categories = []
        subcategories = []

        for news_id in news_ids:
            feat = self._news_features.get(str(news_id), pad)

            titles.append(feat["title"])
            title_masks.append(feat["title_mask"])
            abstracts.append(feat["abstract"])
            abstract_masks.append(feat["abstract_mask"])
            categories.append(feat["category"])
            subcategories.append(feat["subcategory"])

        return {
            "title": torch.tensor(
                [titles],
                dtype=torch.long,
                device=self._device,
            ),
            "title_mask": torch.tensor(
                [title_masks],
                dtype=torch.long,
                device=self._device,
            ),
            "abstract": torch.tensor(
                [abstracts],
                dtype=torch.long,
                device=self._device,
            ),
            "abstract_mask": torch.tensor(
                [abstract_masks],
                dtype=torch.long,
                device=self._device,
            ),
            "category": torch.tensor(
                [categories],
                dtype=torch.long,
                device=self._device,
            ),
            "subcategory": torch.tensor(
                [subcategories],
                dtype=torch.long,
                device=self._device,
            ),
        }

    def _encode_news_vectors(
        self,
        news_ids: list[str],
        batch_size: int = ENCODE_BATCH_SIZE,
    ):
        torch = self._torch

        if self._model is None:
            return None

        valid_ids = [
            str(news_id)
            for news_id in news_ids
            if str(news_id) in self._news_features
        ]

        missing = [
            news_id
            for news_id in valid_ids
            if news_id not in self._news_vector_cache
        ]

        if missing:
            self._model.eval()

            with torch.no_grad():
                for start in range(0, len(missing), batch_size):
                    batch_ids = missing[start:start + batch_size]
                    news_batch = self._news_tensor_batch(batch_ids)

                    vecs = self._model.news_encoder(news_batch).squeeze(0).detach().cpu()

                    for news_id, vec in zip(batch_ids, vecs):
                        self._news_vector_cache[news_id] = vec

        vectors = [
            self._news_vector_cache[news_id]
            for news_id in valid_ids
            if news_id in self._news_vector_cache
        ]

        if not vectors:
            return None

        return torch.stack(vectors).to(self._device)

    def _history_categories(self, history_ids: list[str]) -> tuple[set[str], set[str]]:
        categories: set[str] = set()
        subcategories: set[str] = set()

        if self._news_df is None:
            return categories, subcategories

        for news_id in history_ids:
            idx = self._news_index.get(str(news_id))

            if idx is None:
                continue

            row = self._news_df.iloc[idx]

            category = str(row.get("category") or "").lower().strip()
            subcategory = str(row.get("subcategory") or "").lower().strip()

            if category:
                categories.add(category)

            if subcategory:
                subcategories.add(subcategory)

        return categories, subcategories

    def _candidate_pool(
        self,
        history_ids: list[str],
        max_candidates: int,
        user_id: str | None = None,
    ) -> list[str]:
        if self._news_df is None:
            return []

        valid_ids = set(self._news_index.keys())
        exclude = set(str(x) for x in history_ids)

        candidates: list[str] = []
        seen: set[str] = set()

        if user_id:
            user_id = str(user_id).strip()

            for news_id in self._user_impression_candidates.get(user_id, []):
                if (
                    news_id in valid_ids
                    and news_id not in exclude
                    and news_id not in seen
                ):
                    seen.add(news_id)
                    candidates.append(news_id)

                if len(candidates) >= max_candidates:
                    return candidates[:max_candidates]

        history_categories, history_subcategories = self._history_categories(history_ids)

        if len(candidates) < max_candidates:
            rows = []

            for _, row in self._news_df.iterrows():
                news_id = str(row["news_id"])

                if news_id in exclude or news_id in seen:
                    continue

                category = str(row.get("category") or "").lower().strip()
                subcategory = str(row.get("subcategory") or "").lower().strip()

                subcategory_bonus = 2 if subcategory in history_subcategories else 0
                category_bonus = 1 if category in history_categories else 0
                click_count = self._click_counts.get(news_id, 0)

                rows.append(
                    (
                        news_id,
                        subcategory_bonus,
                        category_bonus,
                        click_count,
                    )
                )

            rows.sort(
                key=lambda x: (
                    x[1],
                    x[2],
                    x[3],
                ),
                reverse=True,
            )

            for news_id, _, _, _ in rows:
                if news_id not in seen:
                    seen.add(news_id)
                    candidates.append(news_id)

                if len(candidates) >= max_candidates:
                    break

        return candidates[:max_candidates]

    def recommend_trained(
        self,
        history_news_ids: list[str],
        k: int = 12,
        max_history: int = 30,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self.recommend(
            history_news_ids=history_news_ids,
            candidate_news_ids=None,
            k=k,
            max_history=max_history,
            max_candidates=DEFAULT_MAX_CANDIDATES,
            user_id=user_id,
        )

    def recommend(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str] | None = None,
        k: int = 12,
        max_history: int = 30,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        self._ensure()

        if not self.available or self._model is None or self._news_df is None:
            return []

        torch = self._torch
        valid_ids = set(self._news_index.keys())

        hist_ids: list[str] = []
        seen_history: set[str] = set()

        for news_id in history_news_ids:
            news_id = str(news_id)

            if news_id in valid_ids and news_id not in seen_history:
                seen_history.add(news_id)
                hist_ids.append(news_id)

        hist_ids = hist_ids[-max_history:]

        if not hist_ids:
            return []

        if candidate_news_ids:
            exclude = set(hist_ids)

            candidate_ids = [
                str(news_id)
                for news_id in candidate_news_ids
                if str(news_id) in valid_ids and str(news_id) not in exclude
            ]

            candidate_ids = candidate_ids[:max_candidates]

        else:
            candidate_ids = self._candidate_pool(
                history_ids=hist_ids,
                max_candidates=max_candidates,
                user_id=user_id,
            )

        if not candidate_ids:
            return []

        self._model.eval()

        with torch.no_grad():
            history_vecs = self._encode_news_vectors(hist_ids)

            candidate_vecs = self._encode_news_vectors(candidate_ids)

            if history_vecs is None or candidate_vecs is None:
                return []

            user_vec = self._model.user_encoder(
                history_vecs.unsqueeze(0)
            ).squeeze(0)

            logits = torch.matmul(
                candidate_vecs,
                user_vec,
            )

            probs = torch.sigmoid(logits).detach().cpu().numpy()

        order = np.argsort(-probs)[:k]

        return [
            (
                str(candidate_ids[int(i)]),
                float(probs[int(i)]),
            )
            for i in order
        ]


_svc: NAMLService | None = None
_svc_lock = threading.Lock()


def get_naml_service(
    root: Path | None = None,
    model_key: str | None = None,
    checkpoint: str = "naml_distilbert_best.pt",
) -> NAMLService:
    global _svc

    with _svc_lock:
        if _svc is None:
            r = root or Path(__file__).resolve().parent

            _svc = NAMLService(
                r,
                checkpoint=checkpoint,
            )

        return _svc