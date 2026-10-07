"""Build results/RESULTS.md (Section 14 of the report) from the real numbers produced by the pipeline.

    python -m src.report
"""
import pandas as pd

from src.utils import load_json, p


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def build() -> str:
    m = load_json(p("results", "metrics.json"))
    if not m:
        raise SystemExit("Run `python -m src.pipeline all` first - results/metrics.json not found")
    ab = load_json(p("results", "ab_result.json"))
    lat = load_json(p("results", "latency.json"))
    drift = load_json(p("results", "drift.json"))
    tune = m.get("hyperparameter_search") or []
    L = ["# Results and Evaluation (auto-generated from real runs)", "",
         f"Run: git `{m['git_commit']}`, data hash `{m['data_hash']}`, test rows {m['test_rows']}, "
         f"relevant = rating >= {m['rating_threshold']}, K = {m['k']}", "",
         "## 1. Model comparison (hold-out test set)", "",
         (p("results") / "results_table.md").read_text(encoding="utf-8"), ""]
    best = m["models"][m["best_model"]]
    pop = m["models"]["popularity"]
    L += [f"**Best model by NDCG@10: `{m['best_model']}`** ({fmt(best['ndcg_at_10'])}) vs popularity baseline "
          f"({fmt(pop['ndcg_at_10'])}) -> "
          f"{'beats' if m['beats_popularity'] else 'does NOT beat'} the baseline.", "",
          "![comparison](model_comparison.png)", ""]
    if tune:
        L += ["## 2. SVD hyper-parameter search (validation set, MLflow nested runs)", "",
              "| n_factors | val RMSE | val MAE | val NDCG@10 |", "|---|---|---|---|"]
        L += [f"| {t['n_factors']} | {t['val_rmse']:.4f} | {t['val_mae']:.4f} | {t['val_ndcg_at_10']:.4f} |"
              for t in tune]
        L.append("")
    seg = p("results", "segment_metrics.csv")
    if seg.exists():
        L += ["## 3. Fairness: NDCG@10 by user segment (best model)", "", "| Segment | Value | NDCG@10 | Users |",
              "|---|---|---|---|"]
        L += [f"| {r.segment} | {r.value} | {r.ndcg_at_10:.4f} | {r.users} |"
              for r in pd.read_csv(seg).itertuples()]
        L.append("")
    if ab:
        L += ["## 4. A/B test (" + ab.get("mode", "") + ")", "",
              f"- Variant A = `{ab['variants']['A']}`, Variant B = `{ab['variants']['B']}`",
              f"- Impressions A/B: {ab.get('impressions_A')} / {ab.get('impressions_B')}; "
              f"clicks A/B: {ab.get('clicks_A')} / {ab.get('clicks_B')}",
              f"- CTR A = {fmt(ab.get('ctr_A'))}, CTR B = {fmt(ab.get('ctr_B'))}, "
              f"relative lift = {fmt(ab.get('relative_lift_B_vs_A'))}",
              f"- z = {fmt(ab.get('z'), 3)}, p-value = {fmt(ab.get('p_value'))} (alpha = {ab.get('alpha')})",
              f"- Needed per variant for a 10% lift: {ab.get('min_sample_per_variant_for_10pct_lift')} impressions "
              f"(enough sample: {ab.get('enough_sample')})",
              f"- **Decision: {ab['decision']}**", ""]
    if lat:
        L += ["## 5. System performance", "",
              f"- p50 = {lat['p50_ms']:.1f} ms, p95 = {lat['p95_ms']:.1f} ms, p99 = {lat['p99_ms']:.1f} ms "
              f"(target p95 < 200 ms: {'MET' if lat['p95_target_met'] else 'NOT met'})",
              f"- Throughput = {lat['throughput_rps']:.1f} req/s, error rate = {lat['error_rate']:.2%}", ""]
    if drift:
        L += ["## 6. Data drift (train window vs newest window)", "", "| Feature | PSI | Drift |", "|---|---|---|"]
        L += [f"| {k} | {v['psi']:.3f} | {v['drift']} |" for k, v in drift["features"].items()]
        L += ["", f"Drift share = {drift['drift_share']:.0%} -> retrain recommended: "
              f"{drift['retrain_recommended']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    text = build()
    (p("results") / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
