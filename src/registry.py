"""Tiny file-based model registry: Staging -> Production -> Archived, with rollback and audit trail."""
import shutil
import time

from src.utils import file_hash, git_commit, load_json, p, save_json


def _path():
    return p("artifacts", "registry.json")


def _load():
    return load_json(_path(), {"versions": [], "production": None, "audit": []})


def production():
    reg = _load()
    return next((v for v in reg["versions"] if v["version"] == reg["production"]), None)


def production_bundle_path():
    prod = production()
    return p("artifacts", "versions", f"v{prod['version']}", "bundle.pkl") if prod else None


def register_candidate(metrics: dict) -> int:
    reg = _load()
    version = len(reg["versions"]) + 1
    dest = p("artifacts", "versions", f"v{version}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(p("artifacts", "candidate", "bundle.pkl"), dest / "bundle.pkl")
    reg["versions"].append({"version": version, "stage": "Staging", "best_model": metrics["best_model"],
                            "ndcg_at_10": metrics["candidate_ndcg"], "git_commit": git_commit(),
                            "sha256": file_hash(dest / "bundle.pkl", 16), "created": int(time.time())})
    reg["audit"].append({"ts": int(time.time()), "action": "register", "version": version})
    save_json(reg, _path())
    return version


def promote(version: int):
    reg = _load()
    for v in reg["versions"]:
        if v["version"] == reg["production"]:
            v["stage"] = "Archived"
        if v["version"] == version:
            v["stage"] = "Production"
    reg["production"] = version
    reg["audit"].append({"ts": int(time.time()), "action": "promote", "version": version})
    save_json(reg, _path())


def rollback():
    reg = _load()
    older = [v for v in reg["versions"] if v["stage"] == "Archived"]
    if not older:
        raise RuntimeError("nothing to roll back to")
    target = max(older, key=lambda v: v["version"])["version"]
    promote(target)
    return target


def better_than_prod(candidate_ndcg: float, min_gain: float = 0.0) -> bool:
    """Champion / challenger quality gate."""
    prod = production()
    return prod is None or candidate_ndcg > prod["ndcg_at_10"] + min_gain
