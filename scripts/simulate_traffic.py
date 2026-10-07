"""Send demo traffic to a RUNNING API so Grafana/Prometheus dashboards and the feedback table have data.

    python scripts/simulate_traffic.py --url http://localhost:8000 --users 300
Clicks are random (5% chance) - this is only for filling dashboards/screenshots, not for reported A/B results.
"""
import argparse
import json
import random
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--max-user", type=int, default=943)
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    for _ in range(a.users):
        uid = random.randint(1, a.max_user)
        rec = json.loads(urllib.request.urlopen(f"{a.url}/recommend/{uid}").read())
        for it in rec["recommendations"]:
            if random.random() < 0.05:
                url = f"{a.url}/feedback?user_id={uid}&item_id={it['item_id']}&event=click&variant={rec['variant']}"
                urllib.request.urlopen(urllib.request.Request(url, method="POST", headers={"x-api-key": a.api_key}))
    print("done")


if __name__ == "__main__":
    main()
