import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB


def _preprocess_X(X_sparse):
    # normalize + log1p
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)
    return X


class Classifier(object):
    def __init__(self):
        self.pipe = make_pipeline(
            GaussianNB(var_smoothing=1e-8)
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        return self.pipe.predict_proba(X)
