# MIND-small loaders: news, behaviors, click-based hot list, category filters.
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _read_behaviors(path: Path) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    return pd.read_csv(path, sep="\t", names=cols, dtype=str)


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
    return pd.read_csv(path, sep="\t", names=cols, dtype=str)


def parse_impressions(cell: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []

    if pd.isna(cell) or not str(cell).strip():
        return out

    for part in str(cell).split():
        if "-" not in part:
            continue

        nid, lbl = part.rsplit("-", 1)

        try:
            out.append((nid, int(lbl)))
        except ValueError:
            continue

    return out


def parse_history(cell: str) -> list[str]:
    if pd.isna(cell) or not str(cell).strip():
        return []

    return [x for x in str(cell).split() if x.startswith("N")]


_IMG_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".svg",
    ".avif",
)


def is_direct_image_url(url: str) -> bool:
    """True if url looks like a direct image asset, not an HTML page."""
    u = (url or "").strip().lower()

    if not u.startswith(("http://", "https://")):
        return False

    base = u.split("?", 1)[0]

    return any(base.endswith(ext) for ext in _IMG_EXTS)


def image_url_for_news(
    url: str,
    title: str,
    width: int = 640,
    height: int = 400,
) -> str:
    """
    Use dataset URL as image if it points to an image file.
    Otherwise return a placeholder image.
    """
    raw = (url or "").strip()

    if raw and raw != "#" and is_direct_image_url(raw):
        return raw

    slug = slug_for_image(title)

    return f"https://placehold.co/{width}x{height}/2a2a2a/eeeeee?text={slug}"


def slug_for_image(title: str, max_len: int = 40) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "+", title)[:max_len].strip("+")
    return t or "News"


class MindRepository:
    """Loads train+val news/behaviors and global click counts from MIND-small."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _project_root()
        self.news_df: pd.DataFrame | None = None
        self._news_index: dict[str, int] = {}

        # Dùng set này để kiểm tra nhanh user đã từng đọc/click bài nào.
        # Không dùng set này để hiển thị lịch sử vì set làm mất thứ tự.
        self.user_clicks: dict[str, set[str]] = defaultdict(set)

        # Danh sách lịch sử đọc theo đúng thứ tự trong behaviors.tsv.
        # Đây là dữ liệu chính dùng cho Reading History và Recommendation.
        self.user_histories_ordered: dict[str, list[str]] = defaultdict(list)

        self.click_counts: Counter[str] = Counter()
        self._user_list: list[str] = []

    def load(self) -> None:
        train_n = self.root / "MINDsmall_train" / "news.tsv"
        val_n = self.root / "MINDsmall_val" / "news.tsv"
        train_b = self.root / "MINDsmall_train" / "behaviors.tsv"
        val_b = self.root / "MINDsmall_val" / "behaviors.tsv"

        for p in (train_n, val_n, train_b, val_b):
            if not p.is_file():
                raise FileNotFoundError(f"Missing MIND file: {p}")

        nt = _read_news(train_n)
        nv = _read_news(val_n)

        news = pd.concat([nt, nv], ignore_index=True)
        news = news.drop_duplicates(subset=["news_id"], keep="first")

        news["title"] = news["title"].fillna("")
        news["abstract"] = news["abstract"].fillna("")
        news["category"] = news["category"].fillna("news").astype(str).str.lower()
        news["subcategory"] = news["subcategory"].fillna("general").astype(str).str.lower()
        news["url"] = news["url"].fillna("#")

        self.news_df = news.reset_index(drop=True)

        self._news_index = {
            nid: i
            for i, nid in enumerate(self.news_df["news_id"].tolist())
        }

        behaviors = pd.concat(
            [
                _read_behaviors(train_b),
                _read_behaviors(val_b),
            ],
            ignore_index=True,
        )

        user_clicks: dict[str, set[str]] = defaultdict(set)

        # Lưu lịch sử theo đúng thứ tự dataset.
        user_histories_ordered: dict[str, list[str]] = defaultdict(list)

        # Dùng để chống trùng nhưng vẫn giữ thứ tự list.
        user_seen: dict[str, set[str]] = defaultdict(set)

        click_counts: Counter[str] = Counter()

        for _, row in behaviors.iterrows():
            uid = str(row["user_id"]).strip()

            if not uid:
                continue

            # 1. Lấy lịch sử đọc thật từ cột history của behaviors.tsv.
            # Đây là lịch sử gốc của user trong dataset MIND.
            for nid in parse_history(row["history"]):
                if nid not in user_seen[uid]:
                    user_seen[uid].add(nid)
                    user_histories_ordered[uid].append(nid)

                user_clicks[uid].add(nid)

            # 2. Lấy thêm các bài user click thật trong impressions, label = 1.
            # Vẫn giữ đúng thứ tự xuất hiện trong behaviors.tsv.
            for nid, lbl in parse_impressions(row["impressions"]):
                if lbl == 1:
                    if nid not in user_seen[uid]:
                        user_seen[uid].add(nid)
                        user_histories_ordered[uid].append(nid)

                    user_clicks[uid].add(nid)
                    click_counts[nid] += 1

        self.user_clicks = user_clicks
        self.user_histories_ordered = user_histories_ordered
        self.click_counts = click_counts
        self._user_list = sorted(user_clicks.keys())

    @property
    def all_user_ids(self) -> list[str]:
        return self._user_list

    def news_row(self, news_id: str) -> pd.Series | None:
        if self.news_df is None:
            return None

        i = self._news_index.get(news_id)

        if i is None:
            return None

        return self.news_df.iloc[i]

    def user_history_ids(self, mind_user_id: str) -> list[str]:
        """
        Return ordered reading history for one MIND user.

        Quan trọng:
        - Không trả về set.
        - Không sorted.
        - Không random.
        - Giữ đúng thứ tự lịch sử đọc trong behaviors.tsv.
        """
        if not mind_user_id:
            return []

        mind_user_id = str(mind_user_id).strip()

        return list(self.user_histories_ordered.get(mind_user_id, []))

    def user_clicked_ids(self, mind_user_id: str) -> set[str]:
        """
        Return clicked/read IDs as a set for fast membership checking only.

        Hàm này chỉ dùng khi cần kiểm tra nhanh một bài có thuộc lịch sử user không.
        Không dùng hàm này để hiển thị Reading History hoặc đưa vào model.
        """
        if not mind_user_id:
            return set()

        mind_user_id = str(mind_user_id).strip()

        return set(self.user_clicks.get(mind_user_id, set()))

    def hot_news_ids(self, k: int = 30) -> list[str]:
        """Most-clicked news in behaviors."""
        ranked = [
            nid
            for nid, _ in self.click_counts.most_common(k * 3)
        ]

        seen: set[str] = set()
        out: list[str] = []

        for nid in ranked:
            if nid in self._news_index and nid not in seen:
                seen.add(nid)
                out.append(nid)

            if len(out) >= k:
                break

        if len(out) < k and self.news_df is not None:
            for nid in self.news_df["news_id"]:
                if nid not in seen:
                    seen.add(nid)
                    out.append(nid)

                if len(out) >= k:
                    break

        return out[:k]

    def category_news_ids(self, category: str, k: int = 60) -> list[str]:
        """
        Return news IDs in one category.

        Example:
            category_news_ids("news", 80)
            category_news_ids("sports", 80)

        Sorting rule:
            1. Articles with more clicks come first.
            2. If click count is equal, keep dataset order.
        """
        if self.news_df is None:
            return []

        category = (category or "").strip().lower()

        if not category:
            return []

        df = self.news_df[
            self.news_df["category"].astype(str).str.lower() == category
        ]

        if df.empty:
            return []

        ids = df["news_id"].tolist()

        # Stable sort:
        # Python sorted là stable, nên nếu click count bằng nhau
        # thì vẫn giữ thứ tự gốc trong ids.
        ids_sorted = sorted(
            ids,
            key=lambda nid: self.click_counts.get(nid, 0),
            reverse=True,
        )

        return ids_sorted[:k]

    def subcategory_news_ids(
        self,
        category: str,
        subcategory: str,
        k: int = 60,
    ) -> list[str]:
        """
        Return news IDs filtered by category and subcategory.
        Useful if later you want /news/newsus, /sports/football_nfl, etc.
        """
        if self.news_df is None:
            return []

        category = (category or "").strip().lower()
        subcategory = (subcategory or "").strip().lower()

        if not category or not subcategory:
            return []

        df = self.news_df[
            (self.news_df["category"].astype(str).str.lower() == category)
            & (self.news_df["subcategory"].astype(str).str.lower() == subcategory)
        ]

        if df.empty:
            return []

        ids = df["news_id"].tolist()

        ids_sorted = sorted(
            ids,
            key=lambda nid: self.click_counts.get(nid, 0),
            reverse=True,
        )

        return ids_sorted[:k]

    def category_counts(self) -> dict[str, int]:
        """
        Return article counts by category.

        Example output:
            {
                "news": 20039,
                "sports": 19368,
                "finance": 3786,
                ...
            }
        """
        if self.news_df is None:
            return {}

        return (
            self.news_df["category"]
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )

    def subcategory_counts(self, category: str | None = None) -> dict[str, int]:
        """
        Return article counts by subcategory.
        If category is provided, count subcategories only inside that category.
        """
        if self.news_df is None:
            return {}

        df = self.news_df

        if category:
            category = category.strip().lower()
            df = df[df["category"].astype(str).str.lower() == category]

        if df.empty:
            return {}

        return (
            df["subcategory"]
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )