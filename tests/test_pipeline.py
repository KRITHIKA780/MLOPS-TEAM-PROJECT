import numpy as np

from src import feedback, pipeline, registry
from src.recommender import ServingModel
from src.utils import load_json, p


def test_end_to_end_pipeline_and_serving(workdir):
    pipeline.ingest(workdir)
    pipeline.validate(workdir)
    pipeline.features(workdir)
    pipeline.train(workdir)
    summary = pipeline.evaluate(workdir)
    assert set(summary["models"]) == {"popularity", "knn", "svd", "content", "hybrid"}
    for m in summary["models"].values():
        assert 0 <= m["ndcg_at_10"] <= 1 and 0.3 < m["rmse"] < 2.5
    assert pipeline.gate(workdir)  # first model always passes (bootstrap)
    v = pipeline.register(workdir)
    assert registry.production()["version"] == v

    import pickle
    model = ServingModel(pickle.load(open(registry.production_bundle_path(), "rb")))
    items, source = model.top_n(1, 10, "svd")
    assert len(items) == 10 and source == "svd"
    seen_ids = set(np.asarray(model.b["iids"])[model.seen[0]])
    assert not seen_ids & {i["item_id"] for i in items}  # never recommend already-seen movies
    cold, source = model.top_n(99999, 10, "svd")  # unknown user -> popularity fallback
    assert source == "popular_fallback" and len(cold) == 10
    assert load_json(p("results", "metrics.json"))["best_model"] in {"knn", "svd", "content", "hybrid"}


def test_feedback_loop_merges_into_training_data(workdir):
    db = p("data", "feedback.db")
    feedback.store_feedback(db, 1, 3, "like")
    feedback.store_feedback(db, 2, 4, "rating", 2.0)
    new = feedback.new_ratings(db)
    assert sorted(new["rating"]) == [2.0, 5.0]
    pipeline.ingest(workdir)
    import pandas as pd
    merged = pd.read_csv(p("data", "processed", "ratings.csv"))
    row = merged[(merged.user_id == 1) & (merged.item_id == 3)]
    assert len(row) == 1 and row["rating"].iloc[0] == 5.0


def test_registry_rollback(workdir):
    pipeline.ingest(workdir)
    pipeline.features(workdir)
    pipeline.train(workdir)
    pipeline.evaluate(workdir)
    v1 = pipeline.register(workdir)
    v2 = pipeline.register(workdir)
    assert registry.production()["version"] == v2
    assert registry.rollback() == v1
