import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier


def _fit_preprocess_P3(X_sparse, top_k=2000):
    X = X_sparse.toarray().astype(np.float32)

    # normalize (row-wise)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    # log1p
    X = np.log1p(X)

    # HVG fit on train
    gene_var = X.var(axis=0)
    gene_idx = np.argsort(gene_var)[-top_k:]
    X = X[:, gene_idx]

    return X.astype(np.float32), gene_idx


def _transform_preprocess_P3(X_sparse, gene_idx):
    X = X_sparse.toarray().astype(np.float32)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    X = np.log1p(X)
    X = X[:, gene_idx]

    return X.astype(np.float32)


class Classifier(object):
    def __init__(self):
        self.top_k = 2000
        self.gene_idx_ = None

        self.le_ = LabelEncoder()
        self.model = None

    def fit(self, X_sparse, y):
        # Encode labels (needed because LGBM uses ints)
        y_enc = self.le_.fit_transform(y)

        # Preprocess P3
        X, gene_idx = _fit_preprocess_P3(X_sparse, top_k=self.top_k)
        self.gene_idx_ = gene_idx

        # Sample weights (balanced)
        sw = compute_sample_weight(class_weight="balanced", y=y_enc)

        # Base models
        base_tree = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )

        bag = BaggingClassifier(
            estimator=base_tree,
            n_estimators=200,
            max_samples=1.0,
            max_features=0.3,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )

        hgb = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=3,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42
        )

        lgbm = LGBMClassifier(
            objective="multiclass",
            num_class=len(self.le_.classes_),
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.6,
            colsample_bytree=0.4,
            reg_lambda=5.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            force_col_wise=True,
            max_bin=127
        )

        # --------------------------
        # Meta-model
        # --------------------------
        final_est = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )

        # Stacking
        self.model = StackingClassifier(
            estimators=[("bag", bag), ("hgb", hgb), ("lgbm", lgbm)],
            final_estimator=final_est,
            stack_method="predict_proba",
            passthrough=False,
            cv=5,
            n_jobs=-1
        )

        # Fit
        self.model.fit(X, y_enc, sample_weight=sw)
        self.classes_ = self.le_.classes_
        return self

    def predict_proba(self, X_sparse):
        X = _transform_preprocess_P3(X_sparse, self.gene_idx_)
        return self.model.predict_proba(X)
