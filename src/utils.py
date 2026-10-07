"""Shared helpers: paths, config, hashing, json."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def home() -> Path:
    """Project home. RECSYS_HOME can redirect all outputs (used by tests)."""
    return Path(os.environ.get("RECSYS_HOME") or REPO)


def p(*parts) -> Path:
    return home().joinpath(*parts)


def load_config(path=None) -> dict:
    cfg_path = Path(path or os.environ.get("RECSYS_CONFIG") or REPO / "configs" / "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        cfg["mlflow"]["tracking_uri"] = uri
    return cfg


def data_dir(cfg) -> Path:
    return p(cfg["data"]["path"])


def feedback_db(cfg) -> Path:
    return p(cfg["serving"]["feedback_db"])


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=REPO, timeout=5)
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def file_hash(path, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
