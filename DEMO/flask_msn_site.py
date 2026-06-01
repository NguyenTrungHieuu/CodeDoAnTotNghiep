from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session, url_for

from mind_backend import MindRepository, image_url_for_news
from users_store import UserStore
from model_registry import get_model_service


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

ROOT = Path(__file__).resolve().parent
user_store = UserStore(ROOT / "data" / "site_users.json")

_repo: MindRepository | None = None
_repo_lock = threading.Lock()

_recommend_cache: dict[str, list[tuple[str, float]]] = {}
_home_nrms_cache: dict[str, list[tuple[str, float]]] = {}
_nrms_page_cache: dict[str, list[tuple[str, float]]] = {}
_dkn_page_cache: dict[str, list[tuple[str, float]]] = {}
_naml_page_cache: dict[str, list[tuple[str, float]]] = {}


CATEGORY_LABELS = {
    "news": "News",
    "sports": "Sports",
    "finance": "Finance",
    "foodanddrink": "Food & Drink",
    "travel": "Travel",
    "lifestyle": "Lifestyle",
    "video": "Video",
    "weather": "Weather",
    "health": "Health",
}

HEADER_CATEGORIES = [
    ("home", "Home", "/"),
    ("news", "News", "/news"),
    ("sports", "Sports", "/sports"),
    ("finance", "Finance", "/finance"),
    ("foodanddrink", "Food & Drink", "/foodanddrink"),
    ("travel", "Travel", "/travel"),
    ("lifestyle", "Lifestyle", "/lifestyle"),
    ("video", "Video", "/video"),
    ("weather", "Weather", "/weather"),
    ("health", "Health", "/health"),
]


def clear_all_recommendation_caches() -> None:
    _recommend_cache.clear()
    _home_nrms_cache.clear()
    _nrms_page_cache.clear()
    _dkn_page_cache.clear()
    _naml_page_cache.clear()


def get_repo() -> MindRepository:
    global _repo

    with _repo_lock:
        if _repo is None:
            repo = MindRepository(ROOT)
            repo.load()
            _repo = repo

    return _repo


def current_user() -> dict | None:
    return session.get("user")


def get_current_mind_user_id() -> str | None:
    user = current_user()

    if not user:
        return None

    mind_user_id = user.get("mind_user_id")

    if not mind_user_id:
        return None

    return str(mind_user_id).strip() or None


def get_user_history_ids(limit: int = 120) -> list[str]:
    user = current_user()

    if not user:
        return []

    repo = get_repo()
    mind_user_id = get_current_mind_user_id()

    mind_history: list[str] = []

    if mind_user_id:
        try:
            mind_history = list(repo.user_history_ids(mind_user_id))
        except Exception as e:
            print(f"Cannot load MIND history for {mind_user_id}: {e}")
            mind_history = []

    session_history = session.get("read_ids", [])

    merged: list[str] = []
    seen: set[str] = set()

    for news_id in mind_history:
        if news_id not in seen:
            seen.add(news_id)
            merged.append(news_id)

    for news_id in session_history:
        if news_id not in seen:
            seen.add(news_id)
            merged.append(news_id)

    return merged[:limit]


def touch_read(news_id: str) -> None:
    if not current_user():
        return

    read_ids = session.get("read_ids", [])

    if news_id in read_ids:
        read_ids.remove(news_id)

    read_ids.insert(0, news_id)
    session["read_ids"] = read_ids[:120]

    clear_all_recommendation_caches()


def article_dict(
    repo: MindRepository,
    news_id: str,
    score: float | None = None,
    badge: str = "Trending",
) -> dict | None:
    row = repo.news_row(news_id)

    if row is None:
        return None

    url = str(row.get("url") or "").strip() or "#"
    abstract = str(row.get("abstract") or "")

    return {
        "news_id": news_id,
        "title": row["title"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "abstract": (abstract[:280] + "…") if len(abstract) > 280 else abstract,
        "url": url,
        "img": image_url_for_news(url, str(row["title"]), 640, 400),
        "score": score,
        "badge": badge,
    }


def article_list_from_ids(
    repo: MindRepository,
    ids: list[str],
    badge: str,
    score: float | None = None,
) -> list[dict]:
    items = [
        article_dict(repo, news_id, score, badge)
        for news_id in ids
    ]

    return [item for item in items if item]


def read_history_list(limit: int = 120) -> list[dict]:
    repo = get_repo()
    history_ids = get_user_history_ids(limit)

    session_ids = set(session.get("read_ids", []))
    mind_user_id = get_current_mind_user_id()

    try:
        mind_ids = set(repo.user_history_ids(mind_user_id)) if mind_user_id else set()
    except Exception:
        mind_ids = set()

    items: list[dict] = []

    for news_id in history_ids:
        row = repo.news_row(news_id)

        if row is None:
            continue

        if news_id in mind_ids:
            source = "MIND user history"
        elif news_id in session_ids:
            source = "Session history"
        else:
            source = "Reading history"

        items.append(
            {
                "news_id": news_id,
                "title": row["title"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "source": source,
            }
        )

    return items


def trending_items(k: int = 30) -> list[dict]:
    repo = get_repo()
    ids = repo.hot_news_ids(k)

    return article_list_from_ids(repo, ids, "Trending")


def trending_feed():
    items = trending_items(30)

    return (
        items[:2],
        items[2:6],
        items[6:14],
        "Trending news",
    )


def build_category_sections() -> dict[str, dict]:
    repo = get_repo()
    sections: dict[str, dict] = {}

    for key, label in CATEGORY_LABELS.items():
        ids = repo.category_news_ids(key, 8)
        items = article_list_from_ids(repo, ids, label)

        sections[key] = {
            "label": label,
            "items": items,
            "url": f"/{key}",
            "count": len(items),
        }

    return sections


def cache_key_for_history(history_ids: list[str], mind_user_id: str | None) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()

    for news_id in history_ids[:50]:
        if news_id not in seen:
            seen.add(news_id)
            cleaned.append(news_id)

    user_part = mind_user_id or "anonymous"

    return user_part + "::" + "|".join(cleaned)


def fallback_recommendations(
    history_ids: list[str],
    k: int = 12,
) -> list[tuple[str, float]]:
    repo = get_repo()
    exclude = set(history_ids)

    candidate_ids = [
        news_id
        for news_id in repo.hot_news_ids(100)
        if news_id not in exclude
    ]

    if not candidate_ids:
        return []

    max_click = 1

    for news_id in candidate_ids:
        max_click = max(max_click, repo.click_counts.get(news_id, 0))

    results: list[tuple[str, float]] = []

    for news_id in candidate_ids[:k]:
        click_count = repo.click_counts.get(news_id, 0)
        normalized = click_count / max_click if max_click > 0 else 0.0
        score = 0.50 + normalized * 0.45

        results.append((news_id, round(float(score), 4)))

    return results


def generate_ai_recommendations(
    history_ids: list[str],
    k: int = 12,
):
    if not history_ids:
        return [], "No reading history found.", None, False

    mind_user_id = get_current_mind_user_id()
    cache_key = cache_key_for_history(history_ids, mind_user_id)

    if cache_key in _recommend_cache:
        return (
            _recommend_cache[cache_key],
            "Loaded cached Co-NAML-LSTUR recommendations.",
            0.0,
            True,
        )

    service = get_model_service(ROOT, "co_naml_lstur")

    if not service.available:
        fallback = fallback_recommendations(history_ids, k)

        return (
            fallback,
            "Co-NAML-LSTUR chưa chạy được nên tạm hiển thị gợi ý fallback theo popularity. "
            f"Chi tiết lỗi: {getattr(service, 'load_error', 'Unknown error')}",
            None,
            False,
        )

    start_time = time.time()

    try:
        recs = service.recommend_trained(
            history_news_ids=history_ids,
            k=k,
            max_history=50,
            user_id=mind_user_id,
        )

        generation_time = time.time() - start_time
        _recommend_cache[cache_key] = recs

        note = (
            service.status_message()
            if hasattr(service, "status_message")
            else "Generated by Co-NAML-LSTUR."
        )

        if mind_user_id:
            note += f" Using MIND user_id: {mind_user_id}."

        return recs, note, generation_time, True

    except TypeError:
        recs = service.recommend_trained(
            history_news_ids=history_ids,
            k=k,
            max_history=50,
        )

        generation_time = time.time() - start_time
        _recommend_cache[cache_key] = recs

        note = "Generated by Co-NAML-LSTUR, but this service does not accept user_id."

        return recs, note, generation_time, True

    except Exception as e:
        fallback = fallback_recommendations(history_ids, k)

        return (
            fallback,
            f"Co-NAML-LSTUR error, showing popularity fallback recommendations: {e}",
            None,
            False,
        )


def generate_home_nrms_recommendations(
    history_ids: list[str],
    k: int = 6,
):
    if not history_ids:
        return [], "NRMS needs reading history.", None, False

    mind_user_id = get_current_mind_user_id()
    cache_key = "home_nrms::" + cache_key_for_history(history_ids, mind_user_id)

    if cache_key in _home_nrms_cache:
        return (
            _home_nrms_cache[cache_key],
            "Loaded cached NRMS DistilBERT home recommendations.",
            0.0,
            True,
        )

    service = get_model_service(ROOT, "nrms_distilbert")

    if not service.available:
        return (
            [],
            f"NRMS DistilBERT unavailable: {getattr(service, 'load_error', 'Unknown error')}",
            None,
            False,
        )

    start_time = time.time()

    try:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
            user_id=mind_user_id,
        )

        generation_time = time.time() - start_time
        _home_nrms_cache[cache_key] = recs

        note = (
            service.status_message()
            if hasattr(service, "status_message")
            else "Generated by NRMS DistilBERT."
        )

        if mind_user_id:
            note += f" Using MIND user_id: {mind_user_id}."

        return recs, note, generation_time, True

    except TypeError:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
        )

        generation_time = time.time() - start_time
        _home_nrms_cache[cache_key] = recs

        return recs, "Generated by NRMS DistilBERT.", generation_time, True

    except Exception as e:
        return [], f"NRMS DistilBERT error: {e}", None, False


def generate_nrms_page_recommendations(
    history_ids: list[str],
    k: int = 12,
):
    if not history_ids:
        return [], "No reading history found for NRMS.", None, False

    mind_user_id = get_current_mind_user_id()
    cache_key = "nrms_page::" + cache_key_for_history(history_ids, mind_user_id)

    if cache_key in _nrms_page_cache:
        return (
            _nrms_page_cache[cache_key],
            "Loaded cached NRMS DistilBERT recommendations.",
            0.0,
            True,
        )

    service = get_model_service(ROOT, "nrms_distilbert")

    if not service.available:
        return (
            [],
            f"NRMS DistilBERT unavailable: {getattr(service, 'load_error', 'Unknown error')}",
            None,
            False,
        )

    start_time = time.time()

    try:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
            user_id=mind_user_id,
        )

        generation_time = time.time() - start_time
        _nrms_page_cache[cache_key] = recs

        note = (
            service.status_message()
            if hasattr(service, "status_message")
            else "Generated by NRMS DistilBERT."
        )

        if mind_user_id:
            note += f" Using MIND user_id: {mind_user_id}."

        return recs, note, generation_time, True

    except TypeError:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
        )

        generation_time = time.time() - start_time
        _nrms_page_cache[cache_key] = recs

        return recs, "Generated by NRMS DistilBERT.", generation_time, True

    except Exception as e:
        return [], f"NRMS DistilBERT error: {e}", None, False


def generate_dkn_page_recommendations(
    history_ids: list[str],
    k: int = 12,
):
    if not history_ids:
        return [], "No reading history found for DKN.", None, False

    mind_user_id = get_current_mind_user_id()
    cache_key = "dkn_page::" + cache_key_for_history(history_ids, mind_user_id)

    if cache_key in _dkn_page_cache:
        return (
            _dkn_page_cache[cache_key],
            "Loaded cached DKN recommendations.",
            0.0,
            True,
        )

    service = get_model_service(ROOT, "dkn")

    if not service.available:
        return (
            [],
            f"DKN unavailable: {getattr(service, 'load_error', 'Unknown error')}",
            None,
            False,
        )

    start_time = time.time()

    try:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
            user_id=mind_user_id,
        )

        generation_time = time.time() - start_time
        _dkn_page_cache[cache_key] = recs

        note = (
            service.status_message()
            if hasattr(service, "status_message")
            else "Generated by DKN."
        )

        if mind_user_id:
            note += f" Using MIND user_id: {mind_user_id}."

        return recs, note, generation_time, True

    except TypeError:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
        )

        generation_time = time.time() - start_time
        _dkn_page_cache[cache_key] = recs

        return recs, "Generated by DKN.", generation_time, True

    except Exception as e:
        return [], f"DKN error: {e}", None, False


def generate_naml_page_recommendations(
    history_ids: list[str],
    k: int = 12,
):
    if not history_ids:
        return [], "No reading history found for NAML.", None, False

    mind_user_id = get_current_mind_user_id()
    cache_key = "naml_page::" + cache_key_for_history(history_ids, mind_user_id)

    if cache_key in _naml_page_cache:
        return (
            _naml_page_cache[cache_key],
            "Loaded cached NAML recommendations.",
            0.0,
            True,
        )

    service = get_model_service(ROOT, "naml")

    if not service.available:
        return (
            [],
            f"NAML unavailable: {getattr(service, 'load_error', 'Unknown error')}",
            None,
            False,
        )

    start_time = time.time()

    try:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
            user_id=mind_user_id,
        )

        generation_time = time.time() - start_time
        _naml_page_cache[cache_key] = recs

        note = (
            service.status_message()
            if hasattr(service, "status_message")
            else "Generated by NAML."
        )

        if mind_user_id:
            note += f" Using MIND user_id: {mind_user_id}."

        return recs, note, generation_time, True

    except TypeError:
        recs = service.recommend_trained(
            history_news_ids=history_ids[-50:],
            k=k,
            max_history=50,
        )

        generation_time = time.time() - start_time
        _naml_page_cache[cache_key] = recs

        return recs, "Generated by NAML.", generation_time, True

    except Exception as e:
        return [], f"NAML error: {e}", None, False


@app.context_processor
def inject_header_categories():
    return {
        "header_categories": HEADER_CATEGORIES,
    }


@app.route("/")
def home():
    repo = get_repo()

    top_stories, grid_stories, sidebar_items, ranking_note = trending_feed()
    category_sections = build_category_sections()

    user = current_user()

    nrms_home_items: list[dict] = []
    nrms_home_note = ""
    nrms_home_time = None
    nrms_home_ok = False

    if user:
        history_ids = get_user_history_ids(500)
        model_history_ids = history_ids[-50:]

        nrms_recs, nrms_home_note, nrms_home_time, nrms_home_ok = (
            generate_home_nrms_recommendations(
                model_history_ids,
                k=6,
            )
        )

        for news_id, score in nrms_recs:
            item = article_dict(
                repo,
                news_id,
                score,
                "NRMS · Click Prediction",
            )

            if item:
                nrms_home_items.append(item)

    return render_template(
        "msn/home.html",
        user=user,
        top_stories=top_stories,
        grid_stories=grid_stories,
        sidebar_items=sidebar_items,
        ranking_note=ranking_note,
        category_sections=category_sections,
        nrms_home_items=nrms_home_items,
        nrms_home_note=nrms_home_note,
        nrms_home_time=nrms_home_time,
        nrms_home_ok=nrms_home_ok,
        active_page="home",
        weather={
            "location": "Tan Lap Commune",
            "temp": 86,
            "condition": "Sunny",
        },
        selected_model="co_naml_lstur",
    )


def render_category_page(category_key: str):
    repo = get_repo()
    category_key = category_key.lower().strip()

    if category_key not in CATEGORY_LABELS:
        flash("Category not found.", "error")
        return redirect(url_for("home"))

    ids = repo.category_news_ids(category_key, 80)
    items = article_list_from_ids(
        repo,
        ids,
        CATEGORY_LABELS[category_key],
    )

    top_stories = items[:2]
    main_items = items[2:18]
    sidebar_items = items[18:30]
    category_counts = repo.category_counts()

    return render_template(
        "msn/category.html",
        user=current_user(),
        category_key=category_key,
        category_label=CATEGORY_LABELS[category_key],
        category_count=category_counts.get(category_key, len(items)),
        total_items=len(items),
        top_stories=top_stories,
        main_items=main_items,
        sidebar_items=sidebar_items,
        active_page=category_key,
    )


@app.route("/news")
def news_page():
    return render_category_page("news")


@app.route("/sports")
def sports_page():
    return render_category_page("sports")


@app.route("/finance")
def finance_page():
    return render_category_page("finance")


@app.route("/foodanddrink")
def foodanddrink_page():
    return render_category_page("foodanddrink")


@app.route("/travel")
def travel_page():
    return render_category_page("travel")


@app.route("/lifestyle")
def lifestyle_page():
    return render_category_page("lifestyle")


@app.route("/video")
def video_page():
    return render_category_page("video")


@app.route("/weather")
def weather_page():
    return render_category_page("weather")


@app.route("/health")
def health_page():
    return render_category_page("health")


@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    user = current_user()

    if not user:
        flash("Please sign in to get personalized AI recommendations.", "error")
        return redirect(url_for("login"))

    repo = get_repo()
    history_ids = get_user_history_ids(120)
    history_items = read_history_list(200)

    recs, ranking_note, generation_time, ai_ok = generate_ai_recommendations(
        history_ids,
        k=12,
    )

    recommendations_list: list[dict] = []

    for news_id, score in recs:
        badge = "AI · Co-NAML-LSTUR" if ai_ok else "Fallback · Popularity"

        item = article_dict(
            repo,
            news_id,
            score,
            badge,
        )

        if item:
            recommendations_list.append(item)

    return render_template(
        "msn/recommendations.html",
        user=user,
        history_items=history_items,
        history_count=len(history_ids),
        recommendations=recommendations_list,
        generated=True,
        generation_time=generation_time,
        ranking_note=ranking_note,
        max_candidates=100,
        selected_model="co_naml_lstur",
        selected_model_label="Co-NAML-LSTUR",
        active_page="ai",
    )


@app.route("/recommendations/", methods=["GET", "POST"])
def recommendations_slash():
    return redirect(url_for("recommendations"))


@app.route("/recommendations-nrms", methods=["GET", "POST"])
def recommendations_nrms():
    user = current_user()

    if not user:
        flash("Please sign in to get NRMS recommendations.", "error")
        return redirect(url_for("login"))

    repo = get_repo()
    history_ids = get_user_history_ids(120)
    history_items = read_history_list(200)

    recs, ranking_note, generation_time, ai_ok = generate_nrms_page_recommendations(
        history_ids,
        k=12,
    )

    recommendations_list: list[dict] = []

    for news_id, score in recs:
        item = article_dict(
            repo,
            news_id,
            score,
            "AI · NRMS DistilBERT" if ai_ok else "NRMS unavailable",
        )

        if item:
            recommendations_list.append(item)

    return render_template(
        "msn/recommendations_nrms.html",
        user=user,
        history_items=history_items,
        history_count=len(history_ids),
        recommendations=recommendations_list,
        generated=True,
        generation_time=generation_time,
        ranking_note=ranking_note,
        max_candidates=100,
        selected_model="nrms_distilbert",
        selected_model_label="NRMS DistilBERT",
        active_page="ai",
    )


@app.route("/recommendations-nrms/", methods=["GET", "POST"])
def recommendations_nrms_slash():
    return redirect(url_for("recommendations_nrms"))


@app.route("/recommendations-dkn", methods=["GET", "POST"])
def recommendations_dkn():
    user = current_user()

    if not user:
        flash("Please sign in to get DKN recommendations.", "error")
        return redirect(url_for("login"))

    repo = get_repo()
    history_ids = get_user_history_ids(120)
    history_items = read_history_list(200)

    recommendations_list: list[dict] = []
    ranking_note = "DKN is ready. Click Regenerate DKN to generate recommendations."
    generation_time = None
    generated = False

    if request.method == "POST":
        recs, ranking_note, generation_time, ai_ok = generate_dkn_page_recommendations(
            history_ids,
            k=12,
        )

        generated = True

        for news_id, score in recs:
            item = article_dict(
                repo,
                news_id,
                score,
                "AI · DKN" if ai_ok else "DKN unavailable",
            )

            if item:
                recommendations_list.append(item)

    return render_template(
        "msn/recommendation_dkn.html",
        user=user,
        history_items=history_items,
        history_count=len(history_ids),
        recommendations=recommendations_list,
        generated=generated,
        generation_time=generation_time,
        ranking_note=ranking_note,
        max_candidates=200,
        selected_model="dkn",
        selected_model_label="DKN",
        active_page="ai",
    )


@app.route("/recommendations-dkn/", methods=["GET", "POST"])
def recommendations_dkn_slash():
    return redirect(url_for("recommendations_dkn"))


@app.route("/recommendations-naml", methods=["GET", "POST"])
def recommendations_naml():
    user = current_user()

    if not user:
        flash("Please sign in to get NAML recommendations.", "error")
        return redirect(url_for("login"))

    repo = get_repo()
    history_ids = get_user_history_ids(120)
    history_items = read_history_list(200)

    recommendations_list: list[dict] = []
    ranking_note = "NAML is ready. Click Regenerate NAML to generate recommendations."
    generation_time = None
    generated = False

    if request.method == "POST":
        recs, ranking_note, generation_time, ai_ok = generate_naml_page_recommendations(
            history_ids,
            k=12,
        )

        generated = True

        for news_id, score in recs:
            item = article_dict(
                repo,
                news_id,
                score,
                "AI · NAML" if ai_ok else "NAML unavailable",
            )

            if item:
                recommendations_list.append(item)

    return render_template(
        "msn/recommendations_naml.html",
        user=user,
        history_items=history_items,
        history_count=len(history_ids),
        recommendations=recommendations_list,
        generated=generated,
        generation_time=generation_time,
        ranking_note=ranking_note,
        max_candidates=300,
        selected_model="naml",
        selected_model_label="NAML",
        active_page="ai",
    )


@app.route("/recommendations-naml/", methods=["GET", "POST"])
def recommendations_naml_slash():
    return redirect(url_for("recommendations_naml"))


@app.route("/reading-history", methods=["GET", "POST"])
def reading_history():
    user = current_user()

    if not user:
        flash("Please sign in to view your reading history.", "error")
        return redirect(url_for("login"))

    if request.method == "POST" and request.form.get("action") == "clear":
        session["read_ids"] = []
        clear_all_recommendation_caches()
        flash("Session reading history cleared.", "info")
        return redirect(url_for("reading_history"))

    return render_template(
        "msn/reading_history.html",
        user=user,
        items=read_history_list(500),
        mind_user_id=user.get("mind_user_id"),
        active_page="history",
    )


@app.route("/reading-history/", methods=["GET", "POST"])
def reading_history_slash():
    return redirect(url_for("reading_history"))


@app.route("/article/<news_id>")
def article(news_id: str):
    repo = get_repo()
    row = repo.news_row(news_id)

    if row is None:
        flash("Article not found.", "error")
        return redirect(url_for("home"))

    touch_read(news_id)

    url = str(row.get("url") or "").strip() or "#"

    return render_template(
        "msn/article.html",
        user=current_user(),
        article={
            "news_id": news_id,
            "title": row["title"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "abstract": row["abstract"],
            "url": url,
            "img": image_url_for_news(url, str(row["title"]), 960, 520),
        },
        active_page=str(row["category"]).lower(),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        user_record = user_store.verify(username, password)

        if user_record:
            session["user"] = user_record
            clear_all_recommendation_caches()
            flash("Signed in successfully.", "success")
            return redirect(url_for("home"))

        flash("Invalid username or password.", "error")

    return render_template(
        "msn/login.html",
        user=current_user(),
        active_page="login",
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    repo = get_repo()
    sample_users = repo.all_user_ids[:400]

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        mind_raw = request.form.get("mind_user_id", "").strip()
        mind_user_id = mind_raw if mind_raw else None

        if mind_user_id and mind_user_id not in repo.user_clicks:
            flash("That MIND user ID was not found in behaviors.tsv.", "error")
        else:
            ok, msg = user_store.register(
                username,
                password,
                mind_user_id,
            )

            if ok:
                flash("Account created. Please sign in.", "success")
                return redirect(url_for("login"))

            flash(msg, "error")

    return render_template(
        "msn/register.html",
        user=current_user(),
        sample_mind_users=sample_users,
        active_page="register",
    )


@app.route("/logout")
def logout():
    session.clear()
    clear_all_recommendation_caches()
    flash("Signed out.", "info")
    return redirect(url_for("home"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    user = current_user()

    if not user:
        flash("Please sign in first.", "error")
        return redirect(url_for("login"))

    repo = get_repo()

    if request.method == "POST":
        if request.form.get("action") == "clear_reads":
            session["read_ids"] = []
            clear_all_recommendation_caches()
            flash("Session reading history cleared.", "info")
            return redirect(url_for("settings"))

        mind_raw = request.form.get("mind_user_id", "").strip()
        mind_user_id = mind_raw if mind_raw else None

        if mind_user_id and mind_user_id not in repo.user_clicks:
            flash("Unknown MIND user ID.", "error")
        else:
            user_store.update_mind_id(
                user["username"],
                mind_user_id,
            )

            session["user"] = {
                "username": user["username"],
                "mind_user_id": mind_user_id,
            }

            clear_all_recommendation_caches()

            flash("Profile updated.", "success")
            return redirect(url_for("settings"))

    return render_template(
        "msn/settings.html",
        user=current_user(),
        sample_mind_users=repo.all_user_ids[:400],
        history_items=read_history_list(200),
        active_page="settings",
    )


@app.route("/sweepstakes")
def sweepstakes():
    return redirect(url_for("home"))


@app.route("/__routes__")
def show_routes():
    routes = [
        str(rule)
        for rule in sorted(
            app.url_map.iter_rules(),
            key=lambda r: str(r),
        )
    ]

    return "<pre>" + "\n".join(routes) + "</pre>"


def open_in_coccoc(url: str) -> None:
    possible_paths = [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "CocCoc"
        / "Browser"
        / "Application"
        / "browser.exe",

        Path("C:/Program Files/CocCoc/Browser/Application/browser.exe"),

        Path("C:/Program Files (x86)/CocCoc/Browser/Application/browser.exe"),
    ]

    for path in possible_paths:
        if path.is_file():
            try:
                subprocess.Popen([str(path), url])
                return
            except Exception:
                pass

    print("Không tìm thấy Cốc Cốc. Hãy mở thủ công:", url)


if __name__ == "__main__":
    print("=" * 60)
    print("MSN-Style News Recommendation System")
    print("=" * 60)

    get_repo()

    co_ckpt = ROOT / "best_co_naml_lstur.pt"
    nrms_ckpt = ROOT / "best_nrms_distilbert.pth"
    dkn_ckpt = ROOT / "dkn_best_model_auc_0.5931.pt"
    naml_ckpt = ROOT / "naml_distilbert_best.pt"

    print("\n📌 Model checkpoints:")
    print(f"   {'✅ Found' if co_ckpt.is_file() else '⚠️ Missing'} Co-NAML-LSTUR: {co_ckpt}")
    print(f"   {'✅ Found' if nrms_ckpt.is_file() else '⚠️ Missing'} NRMS DistilBERT: {nrms_ckpt}")
    print(f"   {'✅ Found' if dkn_ckpt.is_file() else '⚠️ Missing'} DKN: {dkn_ckpt}")
    print(f"   {'✅ Found' if naml_ckpt.is_file() else '⚠️ Missing'} NAML: {naml_ckpt}")

    print("\n📌 Registered Flask routes:")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r)):
        print(f"   {rule}")

    print("\nServer ready: http://127.0.0.1:5000")
    print("=" * 60)

    if os.environ.get("AUTO_OPEN_BROWSER", "1") == "1":
        threading.Timer(
            1.0,
            lambda: open_in_coccoc("http://127.0.0.1:5000"),
        ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )