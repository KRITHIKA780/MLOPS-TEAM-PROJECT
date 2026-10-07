"""Pipeline stages. Used by the CLI, DVC (dvc.yaml) and the Airflow DAG (dags/recsys_training_dag.py).

    python -m src.pipeline all          # ingest -> validate -> features -> train -> evaluate -> gate -> register
    python -m src.pipeline <stage>      # ingest | validate | features | train | evaluate | gate | register | deploy
"""
import json
import os
import sys
import urllib.request

import pandas as pd

from src import registry
from src.data_ingestion import load_items, load_ratings
from src.data_validation import validate_ratings
from src.evaluate import evaluate_candidate
from src.feature_engineering import (build_interaction_features, build_item_features, build_user_features,
                                     popularity_filter, save_table, split_by_time)
from src.train import train_all
from src.utils import load_config, load_json, p, save_json


def ingest(cfg):
    df = popularity_filter(load_ratings(cfg, include_feedback=True))
    (p("data", "processed")).mkdir(parents=True, exist_ok=True)
    df.to_csv(p("data", "processed", "ratings.csv"), index=False)
    print(f"[ingest] {len(df)} ratings saved")


def validate(cfg):
    df = pd.read_csv(p("data", "processed", "ratings.csv"))
    report = validate_ratings(df)  # raises ValidationError -> DAG task fails, production model is untouched
    save_json(report, p("results", "validation.json"))
    print(f"[validate] OK {report['rows']} rows, {report['users']} users, {report['items']} items")


def features(cfg):
    df = pd.read_csv(p("data", "processed", "ratings.csv"))
    items = load_items(cfg)
    tr, va, te = split_by_time(df, cfg["data"]["val_size"], cfg["data"]["test_size"])
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part.to_csv(p("data", "processed", f"{name}.csv"), index=False)
    out = p("data", "processed")
    save_table(build_user_features(tr, items), out / "user_features")
    save_table(build_item_features(tr, items), out / "item_features")
    save_table(build_interaction_features(tr), out / "interaction_features")
    print(f"[features] train={len(tr)} val={len(va)} test={len(te)}")


def train(cfg):
    train_all(cfg)


def evaluate(cfg):
    return evaluate_candidate(cfg)


def gate(cfg) -> bool:
    m = load_json(p("results", "metrics.json"))
    first_model = registry.production() is None  # bootstrap: nothing to compare against yet
    ok = registry.better_than_prod(m["candidate_ndcg"]) and (m["beats_popularity"] or first_model)
    print(f"[gate] candidate NDCG@10={m['candidate_ndcg']:.4f} -> {'PASS' if ok else 'KEEP CURRENT MODEL'}")
    return ok


def register(cfg):
    m = load_json(p("results", "metrics.json"))
    v = registry.register_candidate(m)
    registry.promote(v)
    print(f"[register] model v{v} ({m['best_model']}) promoted to Production")
    return v


def deploy(cfg):
    """Tell the running API to hot-reload the Production bundle."""
    url = os.environ.get("API_URL", "http://localhost:8000") + "/reload"
    try:
        req = urllib.request.Request(url, method="POST", headers={"x-api-key": os.environ.get("API_KEY", "")})
        print("[deploy]", json.loads(urllib.request.urlopen(req, timeout=10).read()))
    except Exception as exc:
        print(f"[deploy] API not reachable ({exc.__class__.__name__}); it loads the Production model on restart")


def run_all(cfg):
    ingest(cfg)
    validate(cfg)
    features(cfg)
    train(cfg)
    evaluate(cfg)
    if gate(cfg):
        register(cfg)
        deploy(cfg)


STAGES = {"ingest": ingest, "validate": validate, "features": features, "train": train, "evaluate": evaluate,
          "gate": gate, "register": register, "deploy": deploy, "all": run_all}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    STAGES[stage](load_config())
