import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def _fit_preprocess_P3(X_sparse, top_k=2000):
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

        base_svm = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )

        # Calibrate to provide predict_proba()
        self.pipe = make_pipeline(
            CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
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
