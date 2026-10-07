"""Stage 1 - Ingestion: download MovieLens-100K and merge new user feedback."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import feedback
from src.synthetic import GENRES
from src.utils import data_dir, feedback_db


def download_movielens(cfg) -> Path:
    target = data_dir(cfg)
    if (target / "u.data").exists():
        return target
    print(f"[ingest] downloading MovieLens-100K from {cfg['data']['url']}")
    raw = urllib.request.urlopen(cfg["data"]["url"], timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(target.parent)  # zip contains the folder 'ml-100k'
    return target


def load_ratings_raw(cfg) -> pd.DataFrame:
    path = download_movielens(cfg)
    return pd.read_csv(path / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])


def load_items(cfg) -> pd.DataFrame:
    path = download_movielens(cfg)
    cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRES
    return pd.read_csv(path / "u.item", sep="|", names=cols, encoding="latin-1")


def load_users(cfg) -> pd.DataFrame:
    path = download_movielens(cfg)
    return pd.read_csv(path / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])


def load_ratings(cfg, include_feedback=True) -> pd.DataFrame:
    """MovieLens ratings + explicit feedback (ratings / likes) collected by the API."""
    df = load_ratings_raw(cfg)
    if include_feedback:
        fb = feedback.new_ratings(feedback_db(cfg))
        if len(fb):
            print(f"[ingest] merging {len(fb)} feedback ratings")
            df = pd.concat([df, fb], ignore_index=True)
            # keep latest rating for each (user, item)
            df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    return df.reset_index(drop=True)
