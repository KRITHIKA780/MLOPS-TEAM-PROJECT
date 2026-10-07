"""Stage 2 - Validation: fail fast on bad data (schema, range, null, duplicates)."""
import pandas as pd


class ValidationError(Exception):
    pass


def validate_ratings(df: pd.DataFrame, min_ratings_per_user: int = 1) -> dict:
    errors = []
    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"missing columns: {missing}")
    if df[required].isnull().any().any():
        errors.append("null values found")
    if len(df) and not df["rating"].between(1, 5).all():
        errors.append("ratings outside 1..5")
    n_dup = int(df.duplicated(["user_id", "item_id"]).sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate (user,item) pairs")
    if (df["user_id"] <= 0).any() or (df["item_id"] <= 0).any():
        errors.append("non-positive ids")
    per_user = df.groupby("user_id").size()
    if len(per_user) and per_user.min() < min_ratings_per_user:
        errors.append(f"users with < {min_ratings_per_user} ratings")
    if len(df) == 0:
        errors.append("empty dataset")
    report = {"rows": int(len(df)), "users": int(df["user_id"].nunique()), "items": int(df["item_id"].nunique()),
              "density": float(len(df) / max(1, df["user_id"].nunique() * df["item_id"].nunique())),
              "mean_rating": float(df["rating"].mean()) if len(df) else None, "errors": errors,
              "passed": not errors}
    if errors:
        raise ValidationError("; ".join(errors))
    return report
