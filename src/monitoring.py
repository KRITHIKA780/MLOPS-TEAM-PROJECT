"""Model/data monitoring: drift (PSI + KS), catalogue coverage and popularity bias."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.synthetic import GENRES


def psi(ref, cur, bins=None, eps=1e-6) -> float:
    """Population Stability Index between two samples (or two aligned distributions if bins is None and 1-D probs)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if bins is not None:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(ref, edges)[0] / max(1, len(ref))
        cur = np.histogram(cur, edges)[0] / max(1, len(cur))
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, items: pd.DataFrame, cfg) -> dict:
    """Compare a reference window with a current window on 4 data features."""
    d = cfg["drift"]
    feats = {}

    def numeric(name, a, b, bins=10):
        stat, pv = ks_2samp(a, b)
        score = psi(a, b, bins=bins)
        feats[name] = {"psi": score, "ks_stat": float(stat), "ks_pvalue": float(pv),
                       "drift": bool(score > d["psi_threshold"] or (pv < d["ks_alpha"] and stat > 0.1))}

    r_ref = ref["rating"].values
    r_cur = cur["rating"].values
    rd = np.arange(1, 6)
    p_ref = np.array([(r_ref == v).mean() for v in rd])
    p_cur = np.array([(r_cur == v).mean() for v in rd])
    stat, pv = ks_2samp(r_ref, r_cur)
    sc = psi(p_ref, p_cur)
    feats["rating_distribution"] = {"psi": sc, "ks_stat": float(stat), "ks_pvalue": float(pv),
                                    "drift": bool(sc > d["psi_threshold"])}
    numeric("user_mean_rating", ref.groupby("user_id")["rating"].mean().values,
            cur.groupby("user_id")["rating"].mean().values)
    numeric("item_mean_rating", ref.groupby("item_id")["rating"].mean().values,
            cur.groupby("item_id")["rating"].mean().values)

    def genre_mix(df):
        m = df.merge(items[["item_id"] + GENRES], on="item_id")
        s = m[GENRES].sum().values.astype(float)
        return s / max(1.0, s.sum())

    gsc = psi(genre_mix(ref), genre_mix(cur))
    feats["genre_mix"] = {"psi": gsc, "drift": bool(gsc > d["psi_threshold"])}
    share = float(np.mean([f["drift"] for f in feats.values()]))
    return {"features": feats, "drift_share": share, "retrain_recommended": bool(share > d["feature_share_threshold"]),
            "rule": f"retrain if > {d['feature_share_threshold']:.0%} of features drift"}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out_html):
    """Optional rich HTML drift report (pip install evidently). Returns False if Evidently is unavailable."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return False
    cols = ["rating", "item_id", "user_id"]
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref[cols], current_data=cur[cols])
    rep.save_html(str(out_html))
    return True


def coverage_and_bias(impr: pd.DataFrame, n_items: int, top_pct=0.01) -> dict:
    """Catalogue coverage and popularity bias of what the system actually showed."""
    if impr.empty:
        return {"coverage": 0.0, "top1pct_share": 0.0}
    counts = impr["item_id"].value_counts()
    top_n = max(1, int(round(top_pct * n_items)))
    return {"coverage": float(counts.size / n_items), "top1pct_share": float(counts.iloc[:top_n].sum() / counts.sum()),
            "alerts": {"coverage_below_20pct": bool(counts.size / n_items < 0.2),
                       "top1pct_over_50pct": bool(counts.iloc[:top_n].sum() / counts.sum() > 0.5)}}
