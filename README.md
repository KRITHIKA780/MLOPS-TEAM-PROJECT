# MLOPS-TEAM-PROJECT: Personalized Movie Recommendation - MLOps Pipeline (Group 6)

End-to-end, continuously improving recommender on **MovieLens 100K**: data ingestion -> validation -> feature
engineering -> training of 3 algorithms (+ popularity baseline + hybrid) -> evaluation -> quality gate -> model
registry -> FastAPI serving with **A/B testing** -> monitoring (Prometheus/Grafana, drift) -> **feedback loop** that
retrains the model.

```
data (DVC) -> ingest -> validate -> features -> train -> evaluate -> quality gate -> register -> deploy (API reload)
                 ^                                                                                     |
                 +---------------- feedback store (impressions, clicks, likes, ratings) <--------------+
```

## Quick start (no Docker)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.pipeline all      # downloads MovieLens-100K, trains, evaluates, registers the best model
python -m src.ab_testing        # A/B test (offline replay) -> results/ab_result.json
uvicorn api.main:app --port 8000
# new terminal:
python -m src.benchmark --url http://localhost:8000 --n 500   # latency / throughput -> results/latency.json
python -m src.report            # builds results/RESULTS.md  (Section 14 of the report)
mlflow ui                       # experiment comparison at http://localhost:5000
```
No internet / just a demo? `python scripts/make_synthetic_data.py` creates a fake dataset (do **not** report those numbers).

## Run each stage separately
`python -m src.pipeline <ingest|validate|features|train|evaluate|gate|register|deploy>`

## Docker
```bash
docker-compose run --rm trainer      # train once (profile "train")
docker-compose up --build            # api :8000, mlflow :5000, prometheus :9090, grafana :3000, airflow :8080
curl http://localhost:8000/recommend/42
```

## API
| Endpoint | Purpose |
|---|---|
| `GET /recommend/{user_id}?n=10` | Top-N list; response includes `variant` (A/B), `model_version`; unknown users get popular items |
| `POST /feedback?user_id=&item_id=&event=click\|like\|dislike\|rating\|watch&rating=` | stored for the feedback loop |
| `GET /health`, `GET /metrics` | health + Prometheus metrics |
| `POST /reload` | hot-reload the Production model (called by the DAG) |
| `GET /ab/results` | live A/B analysis from the feedback DB |

Set `API_KEY` to require the `x-api-key` header on `/feedback` and `/reload`.

## Repository layout
| Path | Purpose |
|---|---|
| `src/data_ingestion.py` | download MovieLens, merge feedback ratings |
| `src/data_validation.py` | schema / range / null / duplicate checks |
| `src/feature_engineering.py` | per-user time split (80/10/10), user/item/interaction features |
| `src/recommender.py` | Popularity, SVD, Item-KNN, Content-based, Hybrid, `ServingModel` |
| `src/train.py` | SVD grid search (validation) + final training -> candidate bundle |
| `src/evaluate.py` | RMSE, MAE, P@10, R@10, NDCG@10, MAP, coverage, segment fairness, drift |
| `src/registry.py` | Staging -> Production -> Archived, rollback, quality gate |
| `src/ab_testing.py` | hash assignment, z-test, sample size, offline replay |
| `src/feedback.py` | SQLite impressions + feedback |
| `src/monitoring.py` | PSI/KS drift, coverage, popularity bias, optional Evidently |
| `src/pipeline.py` | all stages + CLI (used by DVC and Airflow) |
| `api/main.py` | FastAPI service |
| `dags/recsys_training_dag.py` | Airflow DAG |
| `configs/`, `params.yaml`, `dvc.yaml` | configuration and DVC pipeline |
| `tests/` | pytest suite (metrics, split leakage, validation, A/B, end-to-end, feedback loop, rollback) |
| `.github/workflows/ci.yml` | flake8 + pytest + pipeline smoke test + Docker build + Trivy scan |
| `docs/REPORT_GUIDE.md` | where each report number / screenshot comes from |

## Where results are written
`results/metrics.json`, `results/results_table.md`, `results/model_comparison.png`, `results/segment_metrics.csv`,
`results/drift.json`, `results/ab_result.json`, `results/latency.json`, **`results/RESULTS.md`** (combined).

## Design notes / honest limitations
- Models are implemented with numpy/scipy/scikit-learn (no scikit-surprise build problems on Windows).
- The A/B test here is an **offline replay** (a click = recommended movie the user actually liked later in the held-out
  period). Live A/B uses the same code through `/recommend` + `/feedback` + `/ab/results`.
- Impressions of one user are not fully independent; the z-test is therefore optimistic. Report this limitation.
- The model registry is file-based (`artifacts/registry.json`). MLflow tracks runs/metrics/artifacts; with a
  database-backed MLflow server you can additionally register models in the MLflow Model Registry.
- Feedback store uses SQLite; swap the connection for PostgreSQL in production.
