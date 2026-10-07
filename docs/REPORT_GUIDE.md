# How to fill the report from this repo

| Report section | Source in repo |
|---|---|
| 5 Model selection | `configs/config.yaml`, `src/recommender.py` |
| 7 Implementation | `src/`, `api/main.py`, `dags/` (the code in the report is a simplified view of these files) |
| 8 Experiment tracking | `mlflow ui` -> screenshot `screenshots/mlflow_runs.png` (runs: popularity, knn, svd, content, hybrid, svd_hparam_search + nested `svd_nf*`) |
| 9 Workflow | Airflow graph view -> `screenshots/airflow_dag.png`; `dvc.yaml` |
| 10 Serving | `http://localhost:8000/docs` -> `screenshots/api_docs.png` |
| 11 Monitoring | Grafana (queries below) -> `screenshots/grafana.png`; `results/drift.json` |
| 14 Results | **`results/RESULTS.md`** - copy the tables; every `[value]` in the report table comes from `results/results_table.md` |

## Grafana queries (add Prometheus data source http://prometheus:9090)
- Request rate: `sum(rate(rec_requests_total[1m])) by (variant)`
- p95 latency (s): `histogram_quantile(0.95, sum(rate(rec_latency_seconds_bucket[5m])) by (le))`
- Error count: `increase(rec_errors_total[5m])`
- Model version: `rec_model_version`
- Cold-start fallbacks: `increase(rec_cold_start_total[1h])`
- CTR per variant: `sum(rec_feedback_total{event="click"}) / sum(rec_impressions_total)`

## Results discussion template (write after you have real numbers)
1. Which model has the best NDCG@10? Does it beat the popularity baseline? (see `metrics.json: beats_popularity`)
2. RMSE vs ranking quality: a model with lower RMSE is not always better at Top-N ranking - that is why NDCG@10 decides promotion.
3. If personalised models lose to popularity: raise `evaluation.pop_weight` in `configs/config.yaml` (0.3-1.0) and re-run;
   report both settings honestly (pure personalisation vs popularity blend).
4. A/B: CTR A vs B, p-value, sample size vs `min_sample_per_variant_for_10pct_lift`, decision.
5. System: p95 latency vs the 200 ms target, throughput, error rate (`results/latency.json`).
6. Fairness: NDCG@10 by gender / age group / activity (`segment_metrics.csv`) - mention any gap.
7. Drift: which features drifted between the train window and the newest window, and whether retraining is triggered.
