"""Measure API latency/throughput:  python -m src.benchmark --url http://localhost:8000 --n 500"""
import argparse
import json
import random
import time
import urllib.request

import numpy as np

from src.utils import p, save_json


def run(url: str, n: int, max_user: int):
    lat = []
    t0 = time.time()
    errors = 0
    for _ in range(n):
        uid = random.randint(1, max_user)
        s = time.time()
        try:
            urllib.request.urlopen(f"{url}/recommend/{uid}", timeout=5).read()
            lat.append((time.time() - s) * 1000)
        except Exception:
            errors += 1
    total = time.time() - t0
    lat = np.array(lat)
    out = {"requests": n, "errors": errors, "error_rate": errors / n, "p50_ms": float(np.percentile(lat, 50)),
           "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
           "throughput_rps": float(len(lat) / total), "target_p95_ms": 200,
           "p95_target_met": bool(np.percentile(lat, 95) < 200)}
    save_json(out, p("results", "latency.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-user", type=int, default=943)
    a = ap.parse_args()
    print(json.dumps(run(a.url, a.n, a.max_user), indent=2))
