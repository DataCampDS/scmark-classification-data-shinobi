import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def _preprocess_X(X_sparse):
    # Normalize each row by its sum
    X = X_sparse.toarray().astype(np.float32)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    # Log1p transform
    X = np.log1p(X)
    return X


class Classifier(object):
    def __init__(self):
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
