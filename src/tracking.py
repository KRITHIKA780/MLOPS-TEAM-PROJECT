"""Thin MLflow wrapper. If mlflow is not installed, everything becomes a no-op (pipeline still runs)."""
from contextlib import contextmanager
from pathlib import Path

from src.utils import p


class Tracker:
    def __init__(self, cfg):
        self.mlflow = None
        try:
            import mlflow
            uri = cfg["mlflow"]["tracking_uri"]
            if "://" not in uri:
                uri = p(uri).resolve().as_uri()
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(cfg["mlflow"]["experiment"])
            self.mlflow = mlflow
        except Exception as exc:  # mlflow missing or server unreachable
            print(f"[tracking] MLflow disabled ({exc.__class__.__name__}); continuing without tracking")

    @contextmanager
    def run(self, name, tags=None, nested=False):
        if self.mlflow is None:
            yield self
            return
        with self.mlflow.start_run(run_name=name, nested=nested):
            if tags:
                self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield self

    def params(self, d):
        if self.mlflow:
            self.mlflow.log_params({k: str(v) for k, v in d.items()})

    def metrics(self, d):
        if self.mlflow:
            self.mlflow.log_metrics({k: float(v) for k, v in d.items()})

    def artifact(self, path):
        if self.mlflow and Path(path).exists():
            self.mlflow.log_artifact(str(path))
