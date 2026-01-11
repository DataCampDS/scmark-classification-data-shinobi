import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def _preprocess_X_train(X_sparse, top_k=2000):
    
    X = X_sparse.toarray().astype(np.float32)

    # normalize each row by its sum
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    # log1p
    X = np.log1p(X)

    # HVG fit on TRAIN ONLY
    gene_var = X.var(axis=0)
    gene_idx = np.argsort(gene_var)[-top_k:]

    # keep only HVGs
    X = X[:, gene_idx]
    return X, gene_idx


def _preprocess_X_test(X_sparse, gene_idx):
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)

    # apply the SAME HVGs learned from train
    X = X[:, gene_idx]
    return X


class Classifier(object):
    def __init__(self):
        self.top_k = 2000
        self.gene_idx_ = None

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
        X, gene_idx = _preprocess_X_train(X_sparse, top_k=self.top_k)
        self.gene_idx_ = gene_idx

        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _preprocess_X_test(X_sparse, self.gene_idx_)
        return self.pipe.predict_proba(X)
