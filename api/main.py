"""FastAPI serving layer: /recommend, /feedback, /health, /metrics, /reload, /ab/results.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src import ab_testing, feedback, registry
from src.recommender import ServingModel
from src.utils import feedback_db, load_config, p

CFG = load_config()
DB = feedback_db(CFG)
STATE = {"model": None}

REQS = Counter("rec_requests_total", "Recommendation requests", ["variant"])
ERRS = Counter("rec_errors_total", "Failed requests")
IMPR = Counter("rec_impressions_total", "Items shown", ["variant"])
FB = Counter("rec_feedback_total", "Feedback events", ["event"])
COLD = Counter("rec_cold_start_total", "Cold-start fallbacks")
LAT = Histogram("rec_latency_seconds", "Latency of /recommend",
                buckets=(.005, .01, .025, .05, .1, .2, .5, 1, 2))
MODEL_VERSION = Gauge("rec_model_version", "Production model version served")


def load_model() -> bool:
    path = registry.production_bundle_path()
    if path is None or not path.exists():
        fallback = p("artifacts", "candidate", "bundle.pkl")
        path = fallback if fallback.exists() else None
    if path is None:
        return False
    with open(path, "rb") as f:
        bundle = pickle.load(f)  # trusted, self-produced artifact (hash recorded in registry.json)
    prod = registry.production()
    bundle["version"] = prod["version"] if prod else 0
    STATE["model"] = ServingModel(bundle)
    MODEL_VERSION.set(bundle["version"])
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Movie Recommender", version="1.0", lifespan=lifespan)


def require_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    start = time.time()
    model = STATE["model"]
    if model is None:
        ERRS.inc()
        raise HTTPException(status_code=503, detail="model not loaded - run the pipeline first")
    variant = ab_testing.assign_variant(user_id, CFG["ab_test"]["traffic_split_b"])
    model_name = CFG["ab_test"]["variant_b" if variant == "B" else "variant_a"]
    try:
        items, source = model.top_n(user_id, n, model_name)
    except Exception:
        ERRS.inc()
        raise
    if source == "popular_fallback":
        COLD.inc()
    feedback.log_impressions(DB, user_id, variant, [i["item_id"] for i in items])
    REQS.labels(variant).inc()
    IMPR.labels(variant).inc(len(items))
    LAT.observe(time.time() - start)
    return {"user_id": user_id, "variant": variant, "model": source, "model_version": model.version,
            "recommendations": items}


@app.post("/feedback", dependencies=[Depends(require_key)])
def post_feedback(user_id: int, item_id: int, event: str, rating: Optional[float] = None,
                  variant: Optional[str] = None):
    try:
        feedback.store_feedback(DB, user_id, item_id, event, rating,
                                variant or ab_testing.assign_variant(user_id, CFG["ab_test"]["traffic_split_b"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    FB.labels(event).inc()
    return {"status": "saved"}


@app.get("/health")
def health():
    m = STATE["model"]
    return {"status": "ok" if m else "no_model", "model_version": m.version if m else None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload", dependencies=[Depends(require_key)])
def reload_model():
    return {"reloaded": load_model(), "model_version": STATE["model"].version if STATE["model"] else None}


@app.get("/ab/results")
def ab_results():
    return ab_testing.analyse(feedback.ab_counts(DB), CFG["ab_test"]["alpha"])
