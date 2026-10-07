"""Generate a small fake dataset in MovieLens-100K file format (for CI/tests/offline demos only)."""
import numpy as np

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def make_ml100k_like(path, n_users=120, n_items=150, density=0.15, seed=0):
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    k = 4
    U, V = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
    pop = rng.pareto(1.5, n_items) + 0.2
    rows = []
    for u in range(n_users):
        n = max(20, int(density * n_items * rng.uniform(0.5, 1.5)))
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=pop / pop.sum())
        ts = np.sort(rng.integers(880_000_000, 893_000_000, size=len(items)))
        for i, t in zip(items, ts):
            r = int(np.clip(round(3.4 + 0.8 * U[u] @ V[i] / np.sqrt(k) + rng.normal(0, 0.6)), 1, 5))
            rows.append((u + 1, i + 1, r, int(t)))
    with open(path / "u.data", "w") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(path / "u.item", "w", encoding="latin-1") as f:
        for i in range(n_items):
            g = rng.choice(19, size=rng.integers(1, 3), replace=False)
            flags = ["1" if j in g else "0" for j in range(19)]
            yr = int(rng.integers(1960, 1998))
            f.write("|".join([str(i + 1), f"Movie {i + 1} ({yr})", f"01-Jan-{yr}", "", "http://x"] + flags) + "\n")
    with open(path / "u.user", "w") as f:
        for u in range(n_users):
            age, sex = int(rng.integers(15, 65)), rng.choice(["M", "F"])
            f.write("|".join([str(u + 1), str(age), sex, "student", "00000"]) + "\n")
    return path
