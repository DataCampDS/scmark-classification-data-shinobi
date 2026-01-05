import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier


def _preprocess_X_P1(X_sparse):
    # normalize only
    X = X_sparse.toarray().astype(np.float32)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return X / row_sum


class Classifier(object):
    def __init__(self):

        self.pipe = make_pipeline(
                KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
                p=2
            )
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X_P1(X_sparse)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _preprocess_X_P1(X_sparse)
        return self.pipe.predict_proba(X)
