"""Simple JSON user store for demo (non-production)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


class UserStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as f:
                self._users = json.load(f)
        else:
            self._users = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._users, f, indent=2)

    def register(self, username: str, password: str, mind_user_id: str | None) -> tuple[bool, str]:
        u = username.strip()
        if len(u) < 2:
            return False, "Username too short."
        with self._lock:
            self._load()
            if u in self._users:
                return False, "Username already taken."
            self._users[u] = {
                "password_hash": generate_password_hash(password),
                "mind_user_id": (mind_user_id.strip() if mind_user_id else None),
            }
            self._save()
        return True, "OK"

    def verify(self, username: str, password: str) -> dict | None:
        rec = self._users.get(username.strip())
        if not rec:
            return None
        if not check_password_hash(rec["password_hash"], password):
            return None
        return {"username": username.strip(), "mind_user_id": rec.get("mind_user_id")}

    def update_mind_id(self, username: str, mind_user_id: str | None) -> None:
        with self._lock:
            u = self._users.get(username)
            if not u:
                return
            u["mind_user_id"] = mind_user_id.strip() if mind_user_id else None
            self._save()
