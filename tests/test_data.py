import pandas as pd
import pytest

from src.data_ingestion import load_ratings
from src.data_validation import ValidationError, validate_ratings
from src.feature_engineering import split_by_time


def test_split_is_chronological_and_leak_free(workdir):
    df = load_ratings(workdir, include_feedback=False)
    tr, va, te = split_by_time(df, 0.1, 0.1)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(map(tuple, tr[["user_id", "item_id"]].values)) & set(map(tuple, te[["user_id", "item_id"]].values))
    for u, g in te.groupby("user_id"):  # every test rating is newer than (or same time as) the user's train data
        assert g["timestamp"].min() >= tr[tr.user_id == u]["timestamp"].max()


def test_validation_passes_on_clean_data(workdir):
    assert validate_ratings(load_ratings(workdir, include_feedback=False))["passed"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(rating=d["rating"] + 10),
    lambda d: pd.concat([d, d.iloc[:3]]),
    lambda d: d.assign(rating=d["rating"].where(d.index > 0)),
])
def test_validation_fails_on_bad_data(workdir, mutate):
    with pytest.raises(ValidationError):
        validate_ratings(mutate(load_ratings(workdir, include_feedback=False)))
