"""Offline demo / CI: create a fake MovieLens-format dataset in data/raw/ml-100k (NOT for reporting results)."""
from src.synthetic import make_ml100k_like
from src.utils import data_dir, load_config

if __name__ == "__main__":
    print("written to", make_ml100k_like(data_dir(load_config()), n_users=200, n_items=300, density=0.12))
