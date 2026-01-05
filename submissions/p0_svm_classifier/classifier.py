import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def _preprocess_X(X_sparse):
    # P0 preprocessing: raw counts
    return X_sparse.toarray().astype(np.float32)


class Classifier(object):
    def __init__(self):
        base_svm = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=10000,
        )

        # Calibrate to get predict_proba
        self.pipe = make_pipeline(
            CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        return self.pipe.predict_proba(X)
