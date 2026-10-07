"""A/B testing: deterministic variant assignment, two-proportion z-test on CTR, offline replay."""
import hashlib
import math
import pickle

import numpy as np
from scipy.stats import norm

from src import feedback, registry
from src.recommender import ServingModel
from src.utils import p, save_json


def assign_variant(user_id, split_b: float = 0.5) -> str:
    """Hash the user id so every user always sees the same variant."""
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100
    return "B" if h < split_b * 100 else "A"


def two_proportion_ztest(clicks_b, impr_b, clicks_a, impr_a):
    """Two-sided pooled two-proportion z-test (same result as statsmodels proportions_ztest)."""
    p_b, p_a = clicks_b / impr_b, clicks_a / impr_a
    pooled = (clicks_b + clicks_a) / (impr_b + impr_a)
    se = math.sqrt(pooled * (1 - pooled) * (1 / impr_b + 1 / impr_a))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    return z, float(2 * (1 - norm.cdf(abs(z))))


def min_sample_size(p_base: float, rel_lift: float = 0.10, alpha: float = 0.05, power: float = 0.8) -> int:
    """Impressions needed PER VARIANT to detect a relative CTR lift."""
    p1, p2 = p_base, p_base * (1 + rel_lift)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (za * math.sqrt(2 * (p1 + p2) / 2 * (1 - (p1 + p2) / 2)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p2 - p1) ** 2))


def analyse(counts, alpha=0.05) -> dict:
    """counts: DataFrame with columns variant, impressions, clicks."""
    row = {r["variant"]: r for _, r in counts.iterrows()}
    if "A" not in row or "B" not in row:
        return {"decision": "insufficient_data", "reason": "need impressions for both variants"}
    a, b = row["A"], row["B"]
    z, pval = two_proportion_ztest(int(b["clicks"]), int(b["impressions"]), int(a["clicks"]), int(a["impressions"]))
    ctr_a, ctr_b = a["clicks"] / a["impressions"], b["clicks"] / b["impressions"]
    need = min_sample_size(max(ctr_a, 1e-4))
    enough = min(a["impressions"], b["impressions"]) >= need
    promote_b = bool(pval < alpha and ctr_b > ctr_a and enough)
    if promote_b:
        decision = "promote_B"
    elif pval < alpha and ctr_b < ctr_a:
        decision = "rollback_keep_A"
    else:
        decision = "no_significant_difference"
    return {"impressions_A": int(a["impressions"]), "impressions_B": int(b["impressions"]),
            "clicks_A": int(a["clicks"]), "clicks_B": int(b["clicks"]), "ctr_A": float(ctr_a), "ctr_B": float(ctr_b),
            "relative_lift_B_vs_A": float(ctr_b / ctr_a - 1) if ctr_a else None, "z": float(z), "p_value": pval,
            "alpha": alpha, "min_sample_per_variant_for_10pct_lift": need, "enough_sample": bool(enough),
            "decision": decision}


def replay(cfg, bundle_path=None, n=10):
    """OFFLINE REPLAY: serve every test-user with their hash-assigned variant and count a 'click' when a
    recommended item is one the user really liked (rating >= 4) in the held-out test period."""
    path = bundle_path or registry.production_bundle_path() or p("artifacts", "candidate", "bundle.pkl")
    bundle = pickle.load(open(path, "rb"))
    model = ServingModel(bundle)
    rel = bundle["rel_test"]
    ab = cfg["ab_test"]
    models = {"A": ab["variant_a"], "B": ab["variant_b"]}
    db = p("results", "ab_replay.db")
    if db.exists():
        db.unlink()
    for u_idx in np.where(rel.sum(1) > 0)[0]:
        uid = int(bundle["uids"][u_idx])
        var = assign_variant(uid, ab["traffic_split_b"])
        items, _ = model.top_n(uid, n, models[var])
        feedback.log_impressions(db, uid, var, [i["item_id"] for i in items])
        for it in items:
            j = int(np.searchsorted(bundle["iids"], it["item_id"]))
            if rel[u_idx, j]:
                feedback.store_feedback(db, uid, it["item_id"], "click", variant=var)
    result = analyse(feedback.ab_counts(db), ab["alpha"])
    result["variants"] = models
    result["mode"] = "offline replay on held-out test period (not live traffic)"
    save_json(result, p("results", "ab_result.json"))
    return result


if __name__ == "__main__":
    from src.utils import load_config
    print(replay(load_config()))
