import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


def normalize_sum_csr(X):
    X = X.tocsr(copy=True)
    rs = np.asarray(X.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    X = X.multiply((1.0 / rs)[:, None])
    return X


def log1p_csr(X):
    X = X.tocsr(copy=True)
    X.data = np.log1p(X.data)
    return X


def _fit_preprocess_disp_tfidf(X_sparse, top_k=2000, min_df_frac=0.01, max_df_frac=0.95, eps=1e-8):
    # detection count per gene (nnz per column)
    N = X_sparse.shape[0]
    df = np.asarray(X_sparse.getnnz(axis=0)).ravel()

    min_df = int(np.ceil(min_df_frac * N))
    max_df = int(np.floor(max_df_frac * N))

    mask_df = (df >= min_df) & (df <= max_df)
    genes_df_idx = np.where(mask_df)[0]

    # dispersion computed on log1p(normalize_sum)
    X_disp = log1p_csr(normalize_sum_csr(X_sparse[:, genes_df_idx]))

    mu = np.asarray(X_disp.mean(axis=0)).ravel()
    ex2 = np.asarray(X_disp.multiply(X_disp).mean(axis=0)).ravel()
    var = ex2 - mu**2
    var[var < 0] = 0.0

    disp = var / (mu + eps)

    k = min(top_k, disp.size)
    top_local = np.argsort(disp)[-k:]
    gene_idx = genes_df_idx[top_local]

    # TF-IDF on raw counts restricted to selected genes
    X_counts = X_sparse[:, gene_idx].tocsr()
    tfidf = TfidfTransformer(sublinear_tf=True, smooth_idf=True, norm="l2")
    X_tfidf = tfidf.fit_transform(X_counts)

    return X_tfidf, gene_idx, tfidf


def _transform_preprocess_disp_tfidf(X_sparse, gene_idx, tfidf):
    X_counts = X_sparse[:, gene_idx].tocsr()
    return tfidf.transform(X_counts)


class Classifier(object):
    def __init__(self):
        self.top_k = 2000
        self.min_df_frac = 0.01
        self.max_df_frac = 0.95
        self.eps = 1e-8

        self.gene_idx_ = None
        self.tfidf_ = None

        self.model = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        self.classes_ = None

    def fit(self, X_sparse, y):
        X_tfidf, gene_idx, tfidf = _fit_preprocess_disp_tfidf(
            X_sparse,
            top_k=self.top_k,
            min_df_frac=self.min_df_frac,
            max_df_frac=self.max_df_frac,
            eps=self.eps,
        )

        self.gene_idx_ = gene_idx
        self.tfidf_ = tfidf

        self.model.fit(X_tfidf, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X_sparse):
        X_tfidf = _transform_preprocess_disp_tfidf(X_sparse, self.gene_idx_, self.tfidf_)
        return self.model.predict_proba(X_tfidf)
