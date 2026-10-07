import pytest

from src.synthetic import make_ml100k_like
from src.utils import load_config


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Isolated project home with a small synthetic MovieLens-format dataset."""
    monkeypatch.setenv("RECSYS_HOME", str(tmp_path))
    cfg = load_config()
    make_ml100k_like(tmp_path / cfg["data"]["path"], n_users=80, n_items=120, density=0.2)
    return cfg
