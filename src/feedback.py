"""Feedback store (SQLite): impressions + user events (click / like / rating). Feeds the retraining loop."""
import sqlite3
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS impressions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, user_id INTEGER,
    variant TEXT, item_id INTEGER);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, user_id INTEGER,
    item_id INTEGER, event TEXT, rating REAL, variant TEXT);
"""
EVENTS = {"click", "like", "dislike", "rating", "watch"}


def _conn(db):
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.executescript(SCHEMA)
    return con


def log_impressions(db, user_id, variant, item_ids, ts=None):
    ts = int(ts or time.time())
    with _conn(db) as con:
        con.executemany("INSERT INTO impressions (ts,user_id,variant,item_id) VALUES (?,?,?,?)",
                        [(ts, int(user_id), variant, int(i)) for i in item_ids])


def store_feedback(db, user_id, item_id, event, rating=None, variant=None, ts=None):
    if event not in EVENTS:
        raise ValueError(f"event must be one of {sorted(EVENTS)}")
    if event == "rating" and (rating is None or not 1 <= rating <= 5):
        raise ValueError("rating events need rating in 1..5")
    with _conn(db) as con:
        con.execute("INSERT INTO feedback (ts,user_id,item_id,event,rating,variant) VALUES (?,?,?,?,?,?)",
                    (int(ts or time.time()), int(user_id), int(item_id), event, rating, variant))


def new_ratings(db) -> pd.DataFrame:
    """Explicit feedback converted to ratings: rating -> itself, like -> 5, dislike -> 1."""
    cols = ["user_id", "item_id", "rating", "timestamp"]
    if not Path(db).exists():
        return pd.DataFrame(columns=cols)
    with _conn(db) as con:
        df = pd.read_sql_query("SELECT ts AS timestamp, user_id, item_id, event, rating FROM feedback "
                               "WHERE event IN ('rating','like','dislike')", con)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["rating"] = df.apply(lambda r: r["rating"] if r["event"] == "rating" else (5.0 if r["event"] == "like" else 1.0),
                            axis=1)
    return df[cols].astype({"user_id": int, "item_id": int, "timestamp": int})


def ab_counts(db) -> pd.DataFrame:
    """Impressions and clicks per variant."""
    if not Path(db).exists():
        return pd.DataFrame(columns=["variant", "impressions", "clicks"])
    with _conn(db) as con:
        imp = pd.read_sql_query("SELECT variant, COUNT(*) AS impressions FROM impressions GROUP BY variant", con)
        clk = pd.read_sql_query("SELECT variant, COUNT(*) AS clicks FROM feedback WHERE event='click' "
                                "GROUP BY variant", con)
    out = imp.merge(clk, on="variant", how="left").fillna({"clicks": 0})
    out["clicks"] = out["clicks"].astype(int)
    out["ctr"] = out["clicks"] / out["impressions"]
    return out


def impressions_df(db) -> pd.DataFrame:
    if not Path(db).exists():
        return pd.DataFrame(columns=["user_id", "variant", "item_id"])
    with _conn(db) as con:
        return pd.read_sql_query("SELECT user_id, variant, item_id FROM impressions", con)
