"""Stage 4 - Train SVD / Item-KNN / Content-based / Popularity / Hybrid, tune on validation, save candidate bundle."""
import pickle
import time

import numpy as np
import pandas as pd

from src.data_ingestion import load_items
from src.evaluate import ranking_metrics, relevance_matrix, rmse_mae
from src.recommender import (ContentBased, Hybrid, ItemKNN, Popularity, SVDRec, build_index, rank_matrix,
                             to_matrix)
from src.tracking import Tracker
from src.utils import file_hash, git_commit, p, save_json


def train_all(cfg):
    t0 = time.time()
    tr = pd.read_csv(p("data", "processed", "train.csv"))
    va = pd.read_csv(p("data", "processed", "val.csv"))
    te = pd.read_csv(p("data", "processed", "test.csv"))
    allr = pd.concat([tr, va, te])
    items = load_items(cfg)
    idx = build_index(allr, items)
    k, thr = cfg["evaluation"]["top_k"], cfg["evaluation"]["rating_threshold"]
    mc = cfg["models"]
    pw = cfg["evaluation"].get("pop_weight", 1e-4)
    tracker = Tracker(cfg)
    tags = {"git_commit": git_commit(), "data_hash": file_hash(p("data", "processed", "ratings.csv")),
            "stage": "train"}

    # ---- 1) hyper-parameter search for SVD (nested MLflow runs, selected on VALIDATION NDCG@10) ----
    R_tr = to_matrix(tr, idx)
    rel_val = relevance_matrix(va, idx, thr)
    pop_tr = (R_tr > 0).sum(0).astype(float)
    tuning = []
    with tracker.run("svd_hparam_search", tags=tags):
        for nf in mc["svd"]["grid"]:
            with tracker.run(f"svd_nf{nf}", tags=tags, nested=True):
                m = SVDRec(n_factors=nf).fit(R_tr)
                rm, ma = rmse_mae(m.P, va, idx)
                rk = ranking_metrics(rank_matrix(m.P, pop_tr, pw), R_tr > 0, rel_val, k)
                row = {"n_factors": nf, "val_rmse": rm, "val_mae": ma, "val_ndcg_at_10": rk["ndcg_at_10"]}
                tracker.params({"n_factors": nf})
                tracker.metrics({kk: v for kk, v in row.items() if kk != "n_factors"})
                tuning.append(row)
    best_nf = max(tuning, key=lambda r: r["val_ndcg_at_10"])["n_factors"]
    save_json(tuning, p("results", "tuning.json"))
    print(f"[train] SVD search {[(r['n_factors'], round(r['val_ndcg_at_10'], 4)) for r in tuning]} -> best {best_nf}")

    # ---- 2) final fit on train+val, models are scored on the untouched test set in the evaluate stage ----
    R = to_matrix(pd.concat([tr, va]), idx)
    pop = (R > 0).sum(0).astype(float)
    popm = Popularity().fit(R)
    knn = ItemKNN(k=mc["knn"]["k"], min_support=mc["knn"]["min_support"]).fit(R)
    content = ContentBased(max_features=mc["content"]["max_features"], k=mc["content"]["k"]).fit(R, items=items,
                                                                                                  idx=idx)
    svd = SVDRec(n_factors=best_nf).fit(R)
    hybrid = Hybrid(mc["hybrid"]["w_svd"], mc["hybrid"]["w_content"]).fit(svd, content)

    titles = items.set_index("item_id")["title"].reindex(idx.iids).fillna("Unknown").tolist()
    bundle = {
        "uids": idx.uids, "iids": idx.iids, "titles": titles, "pop": pop, "seen": R > 0, "pop_weight": pw,
        "scores": {m.name: m.P.astype(np.float32) for m in [popm, knn, svd, content, hybrid]},
        "params": {"popularity": {}, "knn": mc["knn"], "svd": {"n_factors": best_nf},
                   "content": mc["content"], "hybrid": mc["hybrid"]},
        "meta": {"git_commit": tags["git_commit"], "data_hash": tags["data_hash"], "trained_at": int(time.time()),
                 "train_seconds": round(time.time() - t0, 2)},
    }
    out = p("artifacts", "candidate")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    print(f"[train] candidate bundle saved ({time.time() - t0:.1f}s)")
    return bundle
