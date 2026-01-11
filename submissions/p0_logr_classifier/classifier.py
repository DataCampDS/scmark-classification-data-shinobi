import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def _preprocess_X(X_sparse):
    """
    P0 preprocessing: raw counts
    Input: sparse CSR matrix
    Output: dense numpy array
    """
    return X_sparse.toarray().astype(np.float32)


class Classifier(object):
    def __init__(self):
        # Logistic Regression (no penalty)
        self.pipe = make_pipeline(
            LogisticRegression(
                solver="lbfgs",
                penalty=None,
                max_iter=2000,
                random_state=42,
            )
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        return self.pipe.predict_proba(X)
