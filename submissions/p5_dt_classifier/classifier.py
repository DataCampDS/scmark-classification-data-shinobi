import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


def _fit_preprocess_P5(X_sparse, top_k=2000, n_components=50):
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

    # PCA fit on train
    pca = PCA(n_components=n_components, random_state=0)
    X = pca.fit_transform(X)

    return X, gene_idx, mean, std, pca


def _transform_preprocess_P5(X_sparse, gene_idx, mean, std, pca):
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)

    X = X[:, gene_idx]
    X = (X - mean) / std

    X = pca.transform(X)
    return X


class Classifier(object):
    def __init__(self):
        self.top_k = 2000
        self.n_components = 50

        self.gene_idx_ = None
        self.mean_ = None
        self.std_ = None
        self.pca_ = None

        self.pipe = make_pipeline(
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=5,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
            )
        )

    def fit(self, X_sparse, y):
        X, gene_idx, mean, std, pca = _fit_preprocess_P5(
            X_sparse, top_k=self.top_k, n_components=self.n_components
        )
        self.gene_idx_ = gene_idx
        self.mean_ = mean
        self.std_ = std
        self.pca_ = pca

        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _transform_preprocess_P5(X_sparse, self.gene_idx_, self.mean_, self.std_, self.pca_)
        return self.pipe.predict_proba(X)
