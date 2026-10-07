"""Stage 3 - Time-based split (per user) + user / item / interaction features."""
import pandas as pd

from src.synthetic import GENRES


def split_by_time(df: pd.DataFrame, val_size=0.1, test_size=0.1):
    """Per-user chronological split: oldest 80% train, next 10% validation, newest 10% test (no leakage)."""
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    g = df.groupby("user_id")["item_id"]
    pos = g.cumcount() / g.transform("count")
    train = df[pos < 1 - val_size - test_size]
    val = df[(pos >= 1 - val_size - test_size) & (pos < 1 - test_size)]
    test = df[pos >= 1 - test_size]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def release_year(items: pd.DataFrame) -> pd.Series:
    yr = pd.to_datetime(items["release_date"], errors="coerce", format="%d-%b-%Y").dt.year
    return yr.fillna(0).astype(int)


def year_bucket(year: int) -> str:
    return "year_unknown" if year <= 0 else f"year_{(year // 10) * 10}s"


def build_item_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    stats = train.groupby("item_id")["rating"].agg(item_avg_rating="mean", item_popularity="count").reset_index()
    out = items[["item_id", "title"] + GENRES].copy()
    out["release_year"] = release_year(items)
    out["year_bucket"] = out["release_year"].map(year_bucket)
    out = out.merge(stats, on="item_id", how="left").fillna({"item_avg_rating": 0.0, "item_popularity": 0})
    return out


def build_user_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")["rating"]
    uf = pd.DataFrame({"avg_rating": g.mean(), "rating_count": g.count(), "rating_variance": g.var().fillna(0.0)})
    last = train.groupby("user_id")["timestamp"].max()
    uf["activity_recency_days"] = (train["timestamp"].max() - last) / 86400.0
    merged = train.merge(items[["item_id"] + GENRES], on="item_id")
    # user-genre affinity: mean (rating - user mean) over rated items of that genre
    merged["centered"] = merged["rating"] - merged["user_id"].map(uf["avg_rating"])
    aff = {}
    for gname in GENRES:
        sub = merged[merged[gname] == 1]
        aff[f"aff_{gname}"] = sub.groupby("user_id")["centered"].mean()
    aff = pd.DataFrame(aff).reindex(uf.index).fillna(0.0)
    uf["favourite_genre"] = aff.idxmax(axis=1).str.replace("aff_", "", regex=False)
    return uf.join(aff).reset_index()


def build_interaction_features(train: pd.DataFrame) -> pd.DataFrame:
    out = train.copy()
    out["days_since_rating"] = (out["timestamp"].max() - out["timestamp"]) / 86400.0
    return out


def save_table(df: pd.DataFrame, path_no_ext):
    """Parquet if pyarrow is available, else CSV."""
    try:
        df.to_parquet(f"{path_no_ext}.parquet", index=False)
        return f"{path_no_ext}.parquet"
    except ImportError:
        df.to_csv(f"{path_no_ext}.csv", index=False)
        return f"{path_no_ext}.csv"


def popularity_filter(df: pd.DataFrame, min_user_ratings=1):
    """Minimum-interaction filtering + de-duplication."""
    df = df.drop_duplicates(["user_id", "item_id"], keep="last")
    counts = df.groupby("user_id")["item_id"].transform("count")
    return df[counts >= min_user_ratings]

