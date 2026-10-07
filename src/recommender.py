"""Recommendation models (pure numpy/scipy/sklearn - no heavy install needed).

All models expose:  fit(R, ...) -> self ;  .P  (predicted ratings, users x items, clipped to 1..5) ;
.rank_scores() (scores used for Top-N ranking).
R is a dense user x item matrix with 0 = not rated.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering import release_year, year_bucket
from src.synthetic import GENRES

TIE_BREAK = 1e-4  # tiny popularity bonus so tied scores rank popular items first


class Index:
    def __init__(self, user_ids, item_ids):
        self.uids = np.asarray(sorted(set(map(int, user_ids))))
        self.iids = np.asarray(sorted(set(map(int, item_ids))))
        self.u2i = {u: i for i, u in enumerate(self.uids)}
        self.i2i = {i: j for j, i in enumerate(self.iids)}

    @property
    def shape(self):
        return len(self.uids), len(self.iids)


def build_index(ratings: pd.DataFrame, items: pd.DataFrame) -> Index:
    return Index(ratings["user_id"], list(items["item_id"]) + list(ratings["item_id"]))


def to_matrix(df: pd.DataFrame, idx: Index) -> np.ndarray:
    R = np.zeros(idx.shape)
    if len(df):
        R[df["user_id"].map(idx.u2i).values, df["item_id"].map(idx.i2i).values] = df["rating"].values
    return R


def _baselines(R, item_reg=10.0):
    mask = R > 0
    gm = R.sum() / max(1, mask.sum())
    cnt = mask.sum(1)
    um = np.where(cnt > 0, R.sum(1) / np.maximum(cnt, 1), gm)
    bi = ((R - um[:, None]) * mask).sum(0) / (mask.sum(0) + item_reg)
    return um, bi, gm


class Popularity:
    name = "popularity"

    def fit(self, R, **_):
        mask = R > 0
        um, bi, gm = _baselines(R)
        self.counts = mask.sum(0).astype(float)
        self.P = np.clip(np.tile(gm + bi, (R.shape[0], 1)), 1, 5)
        return self

    def rank_scores(self):
        return np.tile(self.counts, (self.P.shape[0], 1))


class SVDRec:
    """Latent-factor CF: biases + truncated SVD of the residual matrix."""
    name = "svd"

    def __init__(self, n_factors=50, seed=42):
        self.n_factors, self.seed = n_factors, seed

    def fit(self, R, **_):
        mask = R > 0
        um, bi, _ = _baselines(R)
        base = um[:, None] + bi[None, :]
        resid = (R - base) * mask
        k = max(1, min(self.n_factors, min(R.shape) - 1))
        U, s, Vt = svds(csr_matrix(resid), k=k, random_state=self.seed)
        self.P = np.clip((U * s) @ Vt + base, 1, 5)
        return self

    def rank_scores(self):
        return self.P


def _sparsify_topk(S, k):
    S = S.copy()
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    if k < S.shape[1]:
        drop = np.argpartition(S, -k, axis=1)[:, :-k]
        np.put_along_axis(S, drop, 0.0, axis=1)
    return S


def _neighbour_predict(R, S):
    """pred(u,i) = base(u,i) + sum_j S[i,j]*resid(u,j) / sum_j S[i,j]*[u rated j]  (item-based CF)."""
    mask = (R > 0).astype(float)
    um, bi, _ = _baselines(R)
    base = um[:, None] + bi[None, :]
    resid = (R - base) * mask
    num, den = resid @ S.T, mask @ S.T
    adj = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
    return np.clip(base + adj, 1, 5)


class ItemKNN:
    """Item-based KNN with (adjusted) cosine similarity on bias-removed ratings."""
    name = "knn"

    def __init__(self, k=40, min_support=3):
        self.k, self.min_support = k, min_support

    def fit(self, R, **_):
        mask = R > 0
        um, bi, _ = _baselines(R)
        resid = ((R - um[:, None] - bi[None, :]) * mask).T  # items x users
        norm = np.linalg.norm(resid, axis=1, keepdims=True)
        unit = np.divide(resid, norm, out=np.zeros_like(resid), where=norm > 0)
        S = unit @ unit.T
        weak = mask.sum(0) < self.min_support
        S[weak, :] = 0
        S[:, weak] = 0
        self.S = _sparsify_topk(S, self.k)
        self.P = _neighbour_predict(R, self.S)
        return self

    def rank_scores(self):
        return self.P


class ContentBased:
    """TF-IDF over genres + release-decade tokens, cosine item-item similarity, user taste = rated items."""
    name = "content"

    def __init__(self, max_features=500, k=40):
        self.max_features, self.k = max_features, k

    @staticmethod
    def item_docs(items: pd.DataFrame, iids) -> list:
        items = items.set_index("item_id").reindex(iids)
        years = release_year(items.reset_index()).values
        docs = []
        for (_, row), yr in zip(items.iterrows(), years):
            toks = [g.replace("'", "").replace("-", "") for g in GENRES if row.get(g, 0) == 1]
            docs.append(" ".join(toks + [year_bucket(int(yr))]))
        return docs

    def fit(self, R, items=None, idx=None, **_):
        docs = self.item_docs(items, idx.iids)
        X = TfidfVectorizer(max_features=self.max_features, token_pattern=r"\S+").fit_transform(docs)
        S = (X @ X.T).toarray()
        self.S = _sparsify_topk(S, self.k)
        self.P = _neighbour_predict(R, self.S)
        return self

    def rank_scores(self):
        return self.P


class Hybrid:
    """Weighted blend: w_svd * SVD + w_content * content-based."""
    name = "hybrid"

    def __init__(self, w_svd=0.8, w_content=0.2):
        self.w_svd, self.w_content = w_svd, w_content

    def fit(self, svd: SVDRec, content: ContentBased, **_):
        self.P = np.clip(self.w_svd * svd.P + self.w_content * content.P, 1, 5)
        return self

    def rank_scores(self):
        return self.P


def rank_matrix(scores, pop_counts, pop_weight=TIE_BREAK):
    """Ranking scores = predicted rating + pop_weight * normalised popularity.
    Default is a tiny tie-break; raise `evaluation.pop_weight` (e.g. 0.3-1.0) to blend in popularity."""
    pop = pop_counts / max(1.0, pop_counts.max())
    return scores + pop_weight * pop[None, :]


class ServingModel:
    """Loads a trained bundle and produces Top-N lists. Cold-start users get popular items."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.u2i = {int(u): i for i, u in enumerate(bundle["uids"])}
        self.titles = bundle["titles"]
        self.pop = bundle["pop"]
        self.seen = bundle["seen"]
        self.version = bundle.get("version", "unversioned")

    def is_cold(self, user_id: int) -> bool:
        u = self.u2i.get(int(user_id))
        return u is None or not self.seen[u].any()

    def top_n(self, user_id: int, n: int = 10, model: str = "svd"):
        if self.is_cold(user_id):
            order = np.argsort(-self.pop)[:n]
            return [self._item(j, float(self.b["scores"]["popularity"][0, j])) for j in order], "popular_fallback"
        u = self.u2i[int(user_id)]
        w = self.b.get("pop_weight", TIE_BREAK)
        s = rank_matrix(self.b["scores"][model][u:u + 1].astype(float), self.pop, w)[0]
        s[self.seen[u]] = -np.inf
        order = np.argpartition(-s, n)[:n]
        order = order[np.argsort(-s[order])]
        return [self._item(j, float(self.b["scores"][model][u, j])) for j in order], model

    def _item(self, j, score):
        return {"item_id": int(self.b["iids"][j]), "title": self.titles[j], "score": round(score, 3)}
