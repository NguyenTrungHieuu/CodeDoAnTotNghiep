from __future__ import annotations

import json
import threading
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


MAX_TITLE_LEN = 10
MAX_HISTORY_LEN = 50

WORD_EMBED_DIM = 100
ENTITY_EMBED_DIM = 100
NUM_FILTERS = 100
WINDOW_SIZES = [1, 2, 3]


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


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")

    embeddings: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            entity_id = parts[0]

            try:
                vec = np.array(
                    [float(x) for x in parts[1:]],
                    dtype=np.float32,
                )
            except ValueError:
                continue

            embeddings[entity_id] = vec

    if not embeddings:
        raise RuntimeError(f"No entity embeddings found in: {path}")

    return embeddings


def _load_train_entity_embeddings(root: Path) -> dict[str, np.ndarray]:
    train_path = root / "MINDsmall_train" / "entity_embedding.vec"
    val_path = root / "MINDsmall_val" / "entity_embedding.vec"

    if train_path.is_file():
        return _load_embeddings(train_path)

    if val_path.is_file():
        return _load_embeddings(val_path)

    raise FileNotFoundError(
        f"Missing entity_embedding.vec: {train_path} or {val_path}"
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


def _parse_history(cell: str) -> list[str]:
    if pd.isna(cell) or not str(cell).strip():
        return []

    return [
        x
        for x in str(cell).split()
        if x.startswith("N")
    ]


def _parse_title_entities(cell: str) -> list[str]:
    """
    Đúng theo notebook train DKN:
        json.loads(row["TitleEntities"])
        e["EntityID"]

    Không dùng WikidataId ở đây.
    """
    if cell is None:
        return []

    raw = str(cell).strip()

    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    ids: list[str] = []

    for ent in data:
        if not isinstance(ent, dict):
            continue

        entity_id = (
            ent.get("EntityID")
            or ent.get("entity_id")
            or ent.get("EntityId")
            or ent.get("WikidataId")
            or ent.get("WikidataID")
        )

        if entity_id:
            ids.append(str(entity_id))

    return ids


def _load_news_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_news_path = root / "MINDsmall_train" / "news.tsv"
    val_news_path = root / "MINDsmall_val" / "news.tsv"

    if not train_news_path.is_file():
        raise FileNotFoundError(f"Missing file: {train_news_path}")

    if not val_news_path.is_file():
        raise FileNotFoundError(f"Missing file: {val_news_path}")

    train_news = _read_news(train_news_path).fillna("")
    val_news = _read_news(val_news_path).fillna("")

    all_news = pd.concat(
        [train_news, val_news],
        ignore_index=True,
    ).fillna("")

    all_news = all_news.drop_duplicates(
        subset=["news_id"],
        keep="first",
    ).reset_index(drop=True)

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

    behaviors = pd.concat(
        [train_behaviors, val_behaviors],
        ignore_index=True,
    ).fillna("")

    return behaviors


def _build_word_dict_from_train_news(train_news: pd.DataFrame) -> dict[str, int]:
    """
    Đúng theo notebook:
        word_dict = {"[PAD]": 0, "[UNK]": 1}
        for word in title.lower().split()
    """
    word_dict = {
        "[PAD]": 0,
        "[UNK]": 1,
    }

    for title in train_news["title"].fillna("").tolist():
        for word in str(title).lower().split():
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    return word_dict


def _build_entity_dict(entity_embeddings: dict[str, np.ndarray]) -> dict[str, int]:
    """
    Đúng theo notebook:
        entity_dict = {"[PAD]": 0, "[UNK]": 1}
        for eid in entity_emb_dict.keys()
    """
    entity_dict = {
        "[PAD]": 0,
        "[UNK]": 1,
    }

    for entity_id in entity_embeddings.keys():
        entity_dict[str(entity_id)] = len(entity_dict)

    return entity_dict


def _make_entity_matrix(
    entity_dict: dict[str, int],
    entity_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    dim = ENTITY_EMBED_DIM

    for vec in entity_embeddings.values():
        dim = int(vec.shape[0])
        break

    matrix = np.zeros(
        (len(entity_dict), dim),
        dtype=np.float32,
    )

    rng = np.random.default_rng(42)
    matrix[1] = rng.normal(0, 0.01, size=(dim,)).astype(np.float32)

    for entity_id, idx in entity_dict.items():
        if entity_id in ("[PAD]", "[UNK]"):
            continue

        vec = entity_embeddings.get(entity_id)

        if vec is not None and len(vec) == dim:
            matrix[idx] = vec

    return matrix


def _make_news_features(
    news_df: pd.DataFrame,
    word_dict: dict[str, int],
    entity_dict: dict[str, int],
) -> dict[str, tuple[list[int], list[int]]]:
    """
    Đúng preprocessing trong notebook:
        words = row["Title"].lower().split()[:MAX_TITLE_LEN]
        ents = json.loads(row["TitleEntities"])
        e["EntityID"]
    """
    news_features: dict[str, tuple[list[int], list[int]]] = {}

    for _, row in news_df.iterrows():
        news_id = str(row["news_id"])

        title = str(row.get("title") or "")
        title_words = title.lower().split()[:MAX_TITLE_LEN]

        word_ids = [
            word_dict.get(word, 1)
            for word in title_words
        ]

        word_ids = word_ids + [0] * (MAX_TITLE_LEN - len(word_ids))
        word_ids = word_ids[:MAX_TITLE_LEN]

        entity_ids_raw = _parse_title_entities(
            str(row.get("title_entities") or "")
        )[:MAX_TITLE_LEN]

        entity_ids = [
            entity_dict.get(entity_id, 1)
            for entity_id in entity_ids_raw
        ]

        entity_ids = entity_ids + [0] * (MAX_TITLE_LEN - len(entity_ids))
        entity_ids = entity_ids[:MAX_TITLE_LEN]

        news_features[news_id] = (word_ids, entity_ids)

    return news_features


def build_dkn_model():
    torch, nn, F = _load_torch()

    class KCNN(nn.Module):
        def __init__(
            self,
            word_num: int,
            entity_num: int,
            config: dict,
        ):
            super().__init__()

            self.word_embedding = nn.Embedding(
                word_num,
                config["w_dim"],
                padding_idx=0,
            )

            self.entity_embedding = nn.Embedding(
                entity_num,
                config["e_dim"],
                padding_idx=0,
            )

            self.transform_matrix = nn.Parameter(
                torch.empty(
                    config["e_dim"],
                    config["w_dim"],
                ).uniform_(-0.1, 0.1)
            )

            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        2,
                        config["n_filters"],
                        (window, config["w_dim"]),
                    )
                    for window in config["windows"]
                ]
            )

        def forward(self, words, entities):
            word_vec = self.word_embedding(words)
            entity_vec = self.entity_embedding(entities)

            entity_transformed = torch.tanh(
                torch.matmul(entity_vec, self.transform_matrix)
            )

            x = torch.stack(
                [word_vec, entity_transformed],
                dim=1,
            )

            pooled = [
                F.relu(conv(x)).squeeze(3).max(dim=2)[0]
                for conv in self.convs
            ]

            return torch.cat(pooled, dim=1)

    class DKNModel(nn.Module):
        def __init__(
            self,
            word_num: int,
            entity_num: int,
            config: dict,
        ):
            super().__init__()

            self.encoder = KCNN(
                word_num,
                entity_num,
                config,
            )

            feat_dim = len(config["windows"]) * config["n_filters"]

            self.attn = nn.Sequential(
                nn.Linear(feat_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            self.fc = nn.Linear(feat_dim, 1)

        def forward(self, hw, he, cw, ce):
            """
            hw: [B, H, L]
            he: [B, H, L]
            cw: [B, L]
            ce: [B, L]
            """
            batch_size, history_len, title_len = hw.shape

            candidate_vec = self.encoder(cw, ce)

            history_vecs = self.encoder(
                hw.view(-1, title_len),
                he.view(-1, title_len),
            ).view(batch_size, history_len, -1)

            candidate_rep = candidate_vec.unsqueeze(1).expand(
                -1,
                history_len,
                -1,
            )

            attention_input = torch.cat(
                [candidate_rep, history_vecs],
                dim=-1,
            )

            attention_weights = F.softmax(
                self.attn(attention_input).squeeze(-1),
                dim=1,
            )

            user_vec = torch.bmm(
                attention_weights.unsqueeze(1),
                history_vecs,
            ).squeeze(1)

            logits = self.fc(user_vec + candidate_vec).squeeze(-1)

            return logits

    return torch, DKNModel


class DKNService:
    def __init__(
        self,
        root: Path,
        checkpoint: str = "dkn_best_model_auc_0.5931.pt",
    ):
        self.root = root
        self._checkpoint_path = root / checkpoint
        self._lock = threading.Lock()

        self._torch = None
        self._device = None
        self._model = None

        self._train_news: pd.DataFrame | None = None
        self._val_news: pd.DataFrame | None = None
        self._news_df: pd.DataFrame | None = None
        self._behaviors: pd.DataFrame | None = None

        self._news_index: dict[str, int] = {}
        self._news_features: dict[str, tuple[list[int], list[int]]] = {}

        self._word_dict: dict[str, int] = {}
        self._entity_dict: dict[str, int] = {}

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

    def _safe_load(self, torch):
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

    def _extract_state_dict(self, obj):
        if isinstance(obj, dict):
            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "dkn",
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
                "dkn.",
            ]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            cleaned[new_key] = value

        return cleaned

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
                if not self._checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Missing checkpoint: {self._checkpoint_path}"
                    )

                torch, DKNModel = build_dkn_model()

                self._torch = torch
                self._device = torch.device("cpu")

                torch.manual_seed(42)
                np.random.seed(42)

                train_news, val_news, news_df = _load_news_tables(self.root)
                behaviors = _load_behaviors(self.root)

                entity_embeddings = _load_train_entity_embeddings(self.root)

                word_dict = _build_word_dict_from_train_news(train_news)
                entity_dict = _build_entity_dict(entity_embeddings)

                entity_matrix = _make_entity_matrix(
                    entity_dict,
                    entity_embeddings,
                )

                news_features = _make_news_features(
                    news_df,
                    word_dict,
                    entity_dict,
                )

                config = {
                    "w_dim": WORD_EMBED_DIM,
                    "e_dim": ENTITY_EMBED_DIM,
                    "n_filters": NUM_FILTERS,
                    "windows": WINDOW_SIZES,
                }

                model = DKNModel(
                    len(word_dict),
                    len(entity_dict),
                    config,
                ).to(self._device)

                with torch.no_grad():
                    entity_tensor = torch.tensor(
                        entity_matrix,
                        dtype=torch.float32,
                        device=self._device,
                    )

                    if model.encoder.entity_embedding.weight.shape == entity_tensor.shape:
                        model.encoder.entity_embedding.weight.copy_(entity_tensor)

                loaded = self._safe_load(torch)
                state_dict = self._extract_state_dict(loaded)
                state_dict = self._clean_state_dict_keys(state_dict)

                missing, unexpected = model.load_state_dict(
                    state_dict,
                    strict=False,
                )

                model.eval()

                self._missing_keys = len(missing)
                self._unexpected_keys = len(unexpected)

                self._train_news = train_news
                self._val_news = val_news
                self._news_df = news_df
                self._behaviors = behaviors

                self._word_dict = word_dict
                self._entity_dict = entity_dict
                self._news_features = news_features

                self._news_index = {
                    str(news_id): i
                    for i, news_id in enumerate(news_df["news_id"].tolist())
                }

                self._model = model
                self._load_error = None

                self._build_user_candidate_cache()

            except Exception as e:
                self._model = None
                self._train_news = None
                self._val_news = None
                self._news_df = None
                self._behaviors = None
                self._news_index = {}
                self._news_features = {}
                self._load_error = f"{type(e).__name__}: {e}"

    def status_message(self) -> str:
        self._ensure()

        if self._load_error:
            return f"DKN unavailable: {self._load_error}"

        if self._missing_keys == 0 and self._unexpected_keys == 0:
            return (
                f"DKN loaded from `{self._checkpoint_path.name}`. "
                f"All checkpoint keys matched. "
                f"Output is predicted click score."
            )

        return (
            f"DKN loaded from `{self._checkpoint_path.name}`. "
            f"Missing keys: {self._missing_keys}, "
            f"unexpected keys: {self._unexpected_keys}. "
            f"Output is predicted click score."
        )

    def _history_tensors(self, history_ids: list[str]):
        torch = self._torch

        history_ids = history_ids[-MAX_HISTORY_LEN:]

        history_words: list[list[int]] = []
        history_entities: list[list[int]] = []

        for news_id in history_ids:
            words, entities = self._news_features.get(
                str(news_id),
                ([0] * MAX_TITLE_LEN, [0] * MAX_TITLE_LEN),
            )

            history_words.append(words)
            history_entities.append(entities)

        while len(history_words) < MAX_HISTORY_LEN:
            history_words.append([0] * MAX_TITLE_LEN)
            history_entities.append([0] * MAX_TITLE_LEN)

        history_words = history_words[:MAX_HISTORY_LEN]
        history_entities = history_entities[:MAX_HISTORY_LEN]

        hw = torch.tensor(
            [history_words],
            dtype=torch.long,
            device=self._device,
        )

        he = torch.tensor(
            [history_entities],
            dtype=torch.long,
            device=self._device,
        )

        return hw, he

    def _candidate_tensor(self, news_id: str):
        torch = self._torch

        words, entities = self._news_features.get(
            str(news_id),
            ([0] * MAX_TITLE_LEN, [0] * MAX_TITLE_LEN),
        )

        cw = torch.tensor(
            [words],
            dtype=torch.long,
            device=self._device,
        )

        ce = torch.tensor(
            [entities],
            dtype=torch.long,
            device=self._device,
        )

        return cw, ce

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
                if news_id in valid_ids and news_id not in exclude and news_id not in seen:
                    seen.add(news_id)
                    candidates.append(news_id)

        history_categories, history_subcategories = self._history_categories(history_ids)

        if len(candidates) < max_candidates:
            rows = []

            for _, row in self._news_df.iterrows():
                news_id = str(row["news_id"])

                if news_id in exclude or news_id in seen:
                    continue

                category = str(row.get("category") or "").lower().strip()
                subcategory = str(row.get("subcategory") or "").lower().strip()

                category_bonus = 1 if category in history_categories else 0
                subcategory_bonus = 2 if subcategory in history_subcategories else 0
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
        max_history: int = 50,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self.recommend(
            history_news_ids=history_news_ids,
            candidate_news_ids=None,
            k=k,
            max_history=max_history,
            max_candidates=300,
            user_id=user_id,
        )

    def recommend(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str] | None = None,
        k: int = 12,
        max_history: int = 50,
        max_candidates: int = 300,
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

        hw, he = self._history_tensors(hist_ids)

        scores: list[tuple[str, float]] = []

        self._model.eval()

        with torch.no_grad():
            for news_id in candidate_ids:
                cw, ce = self._candidate_tensor(news_id)

                logit = self._model(
                    hw,
                    he,
                    cw,
                    ce,
                )

                prob = torch.sigmoid(logit).item()

                scores.append(
                    (
                        news_id,
                        float(prob),
                    )
                )

        scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return scores[:k]


_svc: DKNService | None = None
_svc_lock = threading.Lock()


def get_dkn_service(
    root: Path | None = None,
    model_key: str | None = None,
    checkpoint: str = "dkn_best_model_auc_0.5931.pt",
) -> DKNService:
    global _svc

    with _svc_lock:
        if _svc is None:
            r = root or Path(__file__).resolve().parent
            _svc = DKNService(
                r,
                checkpoint=checkpoint,
            )

        return _svc