from __future__ import annotations

from pathlib import Path


MODEL_CONFIGS = {
    "co_naml_lstur": {
        "label": "Co-NAML-LSTUR",
        "checkpoint": "best_co_naml_lstur.pt",
        "module": "co_naml_infer",
        "factory": "get_co_naml_service",
    },
    "nrms_distilbert": {
        "label": "NRMS DistilBERT",
        "checkpoint": "best_nrms_distilbert.pth",
        "module": "nrms_infer",
        "factory": "get_nrms_service",
    },
    "dkn": {
        "label": "DKN",
        "checkpoint": "dkn_best_model_auc_0.5931.pt",
        "module": "dkn_infer",
        "factory": "get_dkn_service",
    },
    "naml": {
        "label": "NAML",
        "checkpoint": "naml_distilbert_best.pt",
        "module": "naml_infer",
        "factory": "get_naml_service",
    },
}


DEFAULT_MODEL_KEY = "co_naml_lstur"


class UnavailableModelService:
    available = False

    def __init__(self, label: str, reason: str):
        self.label = label
        self.load_error = str(reason)[:2000]

    def recommend(self, *args, **kwargs):
        return []

    def recommend_trained(self, *args, **kwargs):
        return []

    def status_message(self) -> str:
        return f"{self.label} unavailable: {self.load_error}"


_SERVICE_CACHE: dict[str, object] = {}


def valid_model_key(key: str | None) -> str:
    if key in MODEL_CONFIGS:
        return str(key)

    return DEFAULT_MODEL_KEY


def model_label(key: str | None = None) -> str:
    key = valid_model_key(key)

    return MODEL_CONFIGS[key]["label"]


def model_options(root: Path) -> list[dict]:
    options: list[dict] = []

    for key, cfg in MODEL_CONFIGS.items():
        ckpt = root / cfg["checkpoint"]

        options.append(
            {
                "key": key,
                "label": cfg["label"],
                "checkpoint": str(ckpt),
                "exists": ckpt.is_file(),
            }
        )

    return options


def get_model_service(root: Path, key: str | None = None):
    model_key = valid_model_key(key)
    cfg = MODEL_CONFIGS[model_key]

    root = root.resolve()
    cache_key = f"{root}::{model_key}"

    if cache_key in _SERVICE_CACHE:
        return _SERVICE_CACHE[cache_key]

    label = cfg["label"]
    ckpt = root / cfg["checkpoint"]

    if not ckpt.is_file():
        service = UnavailableModelService(
            label,
            f"Checkpoint not found: {ckpt}",
        )

        _SERVICE_CACHE[cache_key] = service

        return service

    try:
        module = __import__(
            cfg["module"],
            fromlist=[cfg["factory"]],
        )

        factory = getattr(
            module,
            cfg["factory"],
        )

        try:
            service = factory(
                root,
                model_key=model_key,
                checkpoint=cfg["checkpoint"],
            )
        except TypeError:
            service = factory(root)

        _SERVICE_CACHE[cache_key] = service

        return service

    except OSError as e:
        service = UnavailableModelService(
            label,
            "PyTorch DLL load failed. "
            "This is not a model path error. "
            f"Checkpoint exists at: {ckpt}. "
            f"Original error: {type(e).__name__}: {e}",
        )

        _SERVICE_CACHE[cache_key] = service

        return service

    except Exception as e:
        service = UnavailableModelService(
            label,
            f"{type(e).__name__}: {e}",
        )

        _SERVICE_CACHE[cache_key] = service

        return service


def clear_model_cache() -> None:
    _SERVICE_CACHE.clear()