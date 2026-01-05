import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def _fit_preprocess_P4(X_sparse, top_k=2000):
    X = X_sparse.toarray().astype(np.float32)

    # normalize
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    # log1p
    X = np.log1p(X)

    # HVG fit on train
    gene_var = X.var(axis=0)
    gene_idx = np.argsort(gene_var)[-top_k:]
    X = X[:, gene_idx]

    # scaling fit on train
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    return X, gene_idx, mean, std


def _transform_preprocess_P4(X_sparse, gene_idx, mean, std):
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)

    X = X[:, gene_idx]
    X = (X - mean) / std

    return X


class Classifier(object):
    def __init__(self):
        self.top_k = 2000

        self.gene_idx_ = None
        self.mean_ = None
        self.std_ = None

        # Logistic Regression with L2 penalty
        self.pipe = make_pipeline(
            LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=0.05,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        )

    def fit(self, X_sparse, y):
        X, gene_idx, mean, std = _fit_preprocess_P4(X_sparse, top_k=self.top_k)
        self.gene_idx_ = gene_idx
        self.mean_ = mean
        self.std_ = std

        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _transform_preprocess_P4(X_sparse, self.gene_idx_, self.mean_, self.std_)
        return self.pipe.predict_proba(X)
