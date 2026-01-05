import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB


def _fit_preprocess_P3(X_sparse, top_k=2000):
    # normalize + log1p
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)

    # HVG fit on train
    gene_var = X.var(axis=0)
    gene_idx = np.argsort(gene_var)[-top_k:]
    X = X[:, gene_idx]

    return X, gene_idx


def _transform_preprocess_P3(X_sparse, gene_idx):
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)

    X = X[:, gene_idx]
    return X


class Classifier(object):
    def __init__(self):
        self.top_k = 2000
        self.gene_idx_ = None

        self.pipe = make_pipeline(
            GaussianNB(var_smoothing=1e-7)
        )

    def fit(self, X_sparse, y):
        X, gene_idx = _fit_preprocess_P3(X_sparse, top_k=self.top_k)
        self.gene_idx_ = gene_idx

        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _transform_preprocess_P3(X_sparse, self.gene_idx_)
        return self.pipe.predict_proba(X)
