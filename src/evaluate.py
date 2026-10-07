"""Metrics + Stage 5 (evaluate candidate bundle on the hold-out test set)."""
import pickle

import numpy as np
import pandas as pd

from src import monitoring
from src.data_ingestion import load_items, load_users
from src.recommender import rank_matrix
from src.tracking import Tracker
from src.utils import file_hash, git_commit, load_json, p, save_json


def rmse_mae(P, df, idx):
    pred = P[df["user_id"].map(idx.u2i).values, df["item_id"].map(idx.i2i).values]
    err = pred - df["rating"].values
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def relevance_matrix(df, idx, threshold=4.0):
    rel = np.zeros(idx.shape, dtype=bool)
    good = df[df["rating"] >= threshold]
    rel[good["user_id"].map(idx.u2i).values, good["item_id"].map(idx.i2i).values] = True
    return rel


def ranking_metrics(scores, seen, rel, k=10, per_user=False):
    """Precision@k, Recall@k, NDCG@k, MAP@k, catalogue coverage. Items in `seen` are never recommended."""
    s = np.where(seen, -np.inf, scores)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    users = np.where(rel.sum(1) > 0)[0]
    hits = np.take_along_axis(rel, top, axis=1)[users].astype(float)
    n_rel = rel[users].sum(1).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    idcg = np.array([disc[:int(min(n, k))].sum() for n in n_rel])
    prec_at = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    ap = (prec_at * hits).sum(1) / np.minimum(n_rel, k)
    out = {"precision_at_10": float(hits.mean()), "recall_at_10": float((hits.sum(1) / n_rel).mean()),
           "ndcg_at_10": float((dcg / idcg).mean()), "map_at_10": float(ap.mean()),
           "coverage": float(len(np.unique(top[users])) / scores.shape[1]), "users_evaluated": int(len(users))}
    if per_user:
        out["_users"], out["_ndcg"] = users, dcg / idcg
    return out


def segment_metrics(users_idx, ndcg, idx, users_df, train_counts):
    """NDCG@10 by gender / age group / activity level (fairness check)."""
    df = pd.DataFrame({"user_id": idx.uids[users_idx], "ndcg": ndcg}).merge(users_df, on="user_id", how="left")
    df["age_group"] = pd.cut(df["age"], [0, 24, 34, 49, 120], labels=["<25", "25-34", "35-49", "50+"])
    cnt = pd.Series(train_counts, index=idx.uids)
    df["activity"] = pd.qcut(df["user_id"].map(cnt).rank(method="first"), 3, labels=["light", "medium", "heavy"])
    rows = []
    for col in ["gender", "age_group", "activity"]:
        g = df.groupby(col, observed=True)["ndcg"].agg(["mean", "count"]).reset_index()
        g.columns = ["value", "ndcg_at_10", "users"]
        g.insert(0, "segment", col)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def evaluate_candidate(cfg):
    k, thr = cfg["evaluation"]["top_k"], cfg["evaluation"]["rating_threshold"]
    pw = cfg["evaluation"].get("pop_weight", 1e-4)
    b = pickle.load(open(p("artifacts", "candidate", "bundle.pkl"), "rb"))
    from src.recommender import Index
    idx = Index(b["uids"], b["iids"])
    train = pd.read_csv(p("data", "processed", "train.csv"))
    val = pd.read_csv(p("data", "processed", "val.csv"))
    test = pd.read_csv(p("data", "processed", "test.csv"))
    rel = relevance_matrix(test, idx, thr)
    b["rel_test"] = rel
    pickle.dump(b, open(p("artifacts", "candidate", "bundle.pkl"), "wb"), protocol=4)

    tracker = Tracker(cfg)
    tags = {"git_commit": git_commit(), "data_hash": file_hash(p("data", "processed", "ratings.csv")),
            "stage": "evaluate"}
    results = {}
    for name, sc in b["scores"].items():
        P = sc.astype(float)
        rmse, mae = rmse_mae(P, test, idx)
        rank_sc = np.tile(b["pop"], (P.shape[0], 1)) if name == "popularity" else P
        m = ranking_metrics(rank_matrix(rank_sc, b["pop"], pw), b["seen"], rel, k)
        m.update({"rmse": rmse, "mae": mae})
        results[name] = m
        with tracker.run(name, tags=tags):
            tracker.params(b["params"].get(name, {}))
            tracker.metrics({kk: v for kk, v in m.items()})
    contenders = {n: m["ndcg_at_10"] for n, m in results.items() if n != "popularity"}
    best = max(contenders, key=contenders.get)

    # fairness / segment analysis for the best model
    users_df = load_users(cfg)
    bm = ranking_metrics(rank_matrix(b["scores"][best].astype(float), b["pop"], pw), b["seen"], rel, k,
                         per_user=True)
    seg = segment_metrics(bm["_users"], bm["_ndcg"], idx, users_df, (train.groupby("user_id").size()
                          .reindex(idx.uids).fillna(0).values))
    seg.to_csv(p("results", "segment_metrics.csv"), index=False)

    # drift between training window and newest window
    drift = monitoring.compute_drift(pd.concat([train, val]), test, load_items(cfg), cfg)
    save_json(drift, p("results", "drift.json"))

    summary = {"models": results, "best_model": best, "candidate_ndcg": contenders[best],
               "popularity_ndcg": results["popularity"]["ndcg_at_10"],
               "beats_popularity": contenders[best] > results["popularity"]["ndcg_at_10"],
               "k": k, "rating_threshold": thr, "test_rows": int(len(test)), "git_commit": tags["git_commit"],
               "data_hash": tags["data_hash"], "hyperparameter_search": load_json(p("results", "tuning.json"), [])}
    save_json(summary, p("results", "metrics.json"))
    write_results_table(summary)
    plot_comparison(summary)
    with tracker.run("summary", tags=tags):
        tracker.metrics({"best_ndcg_at_10": summary["candidate_ndcg"]})
        for f in ["metrics.json", "results_table.md", "model_comparison.png", "segment_metrics.csv", "drift.json"]:
            tracker.artifact(p("results", f))
    print(f"[evaluate] best={best} NDCG@10={summary['candidate_ndcg']:.4f} "
          f"(popularity={summary['popularity_ndcg']:.4f})")
    return summary


def write_results_table(summary):
    rows = ["| Model | RMSE | MAE | P@10 | R@10 | NDCG@10 | MAP@10 | Coverage |", "|---|---|---|---|---|---|---|---|"]
    label = {"popularity": "Popularity baseline", "knn": "Item-based KNN", "svd": "SVD", "content": "Content-based",
             "hybrid": "Hybrid (0.8 SVD + 0.2 content)"}
    for name in ["popularity", "knn", "svd", "content", "hybrid"]:
        m = summary["models"][name]
        rows.append(f"| {label[name]} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['precision_at_10']:.4f} | "
                    f"{m['recall_at_10']:.4f} | {m['ndcg_at_10']:.4f} | {m['map_at_10']:.4f} | {m['coverage']:.3f} |")
    (p("results") / "results_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_comparison(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(summary["models"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, [summary["models"][n]["rmse"] for n in names], color="#4C78A8")
    ax[0].set_title("RMSE (lower is better)")
    w = 0.27
    x = np.arange(len(names))
    for i, (key, lab) in enumerate([("precision_at_10", "P@10"), ("recall_at_10", "R@10"), ("ndcg_at_10", "NDCG@10")]):
        ax[1].bar(x + (i - 1) * w, [summary["models"][n][key] for n in names], w, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_title("Ranking quality @10 (higher is better)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(p("results", "model_comparison.png"), dpi=130)
    plt.close(fig)
