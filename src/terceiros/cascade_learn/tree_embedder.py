from numbers import Real, Integral
import warnings

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.tree
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
    _fit_context,
)
from sklearn.ensemble._forest import BaseForest
from sklearn.tree._classes import BaseDecisionTree, DTYPE
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import (
    validate_params,
    StrOptions,
    Interval,
)

__all__ = [
    "embed_with_tree",
    "TreeEmbedder",
    "ForestEmbedder",
]

_COMMON_PARAMS = {
    "method": [StrOptions({"all_nodes", "path", "dense_path"})],
    "node_weights": [
        StrOptions({"neg_log_frac", "log_node_size", "norm_log_node_size"}),
        callable,
        None,
    ],
    "max_pvalue": [
        Interval(Real, 0, 1, closed="both"),
    ],
    "max_node_size": [
        Interval(Real, 0, 1, closed="right"),
        Interval(Integral, 1, None, closed="left"),
    ],
}


def _chi2_per_node(
    tree: sklearn.tree._tree.Tree,
    yates_correction=False,
    warn: bool = True,
):
    if tree.value.ndim == 3:
        class_counts = tree.value
    else:
        class_counts = np.dstack([
            (1 - tree.value) * tree.weighted_n_node_samples,
            tree.value * tree.weighted_n_node_samples,
        ])

    margin_row = class_counts[0]
    total = tree.weighted_n_node_samples[0]

    margin_col = np.stack([
        tree.weighted_n_node_samples[1:],
        total - tree.weighted_n_node_samples[1:],
    ]).T

    expected = (margin_row[None, :, None, :] * margin_col[:, None, :, None]) / total

    if 0. in expected:
        if warn:
            warnings.warn(
                "It seems that some classes are not observed in the dataset received by"
                " the tree. The following [y_column, class_index] are not found in y:"
                f" {np.argwhere(margin_row == 0).tolist()}"
            )
        expected[expected == 0] = np.finfo(expected.dtype).eps

    contingency_tables = np.stack(
        [
            class_counts[1:],
            margin_row - class_counts[1:],
        ],
        axis=-2,
    )

    diff = np.abs(contingency_tables - expected)
    if yates_correction:
        diff[diff <= 0.5] = 0.0
        diff[diff > 0.5] -= 0.5

    chi2 = (diff ** 2 / expected).sum(axis=(-1, -2))

    return np.concatenate([
        np.zeros((1, tree.n_outputs), dtype=chi2.dtype),
        chi2,
    ])


def _hstack(Xs):
    if any(scipy.sparse.issparse(f) for f in Xs):
        return scipy.sparse.hstack(Xs, format='csr')
    return np.hstack(Xs)


@validate_params(
    {
        "node_size": ["array-like"],
        "node_weights": _COMMON_PARAMS["node_weights"],
    },
    prefer_skip_nested_validation=True,
)
def _get_node_weights(node_size, node_weights):
    if callable(node_weights):
        return node_weights(node_size)
    if node_weights == "neg_log_frac":
        return -np.log(node_size / node_size[0])
    if node_weights == "log_node_size":
        return 1 / (np.log(node_size) + 1)
    if node_weights == "norm_log_node_size":
        log_size = np.log(node_size)
        return (1 - log_size / np.log(node_size[0])) / (1 + log_size)
    if node_weights is None:
        return np.ones(node_size.shape)
    raise RuntimeError


@validate_params(
    {
        "tree": [BaseDecisionTree],
        "X": ["array-like"],
        "warn_no_output": ["boolean"],
        **_COMMON_PARAMS,
    },
    prefer_skip_nested_validation=False,
)
def embed_with_tree(
    tree_estimator,
    X,
    method="path",
    node_weights=None,
    max_pvalue=1.0,
    max_node_size=1.0,
    warn_no_output=True,
) -> np.ndarray | scipy.sparse.csr_matrix:
    tree = tree_estimator.tree_
    original_max_node_size = max_node_size

    if method != "dense_path":
        mask = np.ones(tree.node_count, dtype=bool)

        if max_pvalue < 1.0:
            chi2 = _chi2_per_node(tree)
            chi2 = chi2.max(axis=1)
            chi2_threshold = scipy.stats.chi2.ppf(max_pvalue, df=1)
            mask &= chi2 > chi2_threshold

        node_size = tree.weighted_n_node_samples

        if isinstance(max_node_size, float):
            max_node_size = np.ceil(max_node_size * node_size[0])

        mask &= node_size <= max_node_size

        if method == "all_nodes":
            mask &= tree.children_left != tree.children_right

        weights = _get_node_weights(node_size[mask], node_weights)

    if method == "path":
        Xt = tree_estimator.decision_path(X)[:, mask]

        if node_weights is not None:
            Xt = Xt.multiply(weights).tocsr()

    elif method == "dense_path":
        if node_weights is not None:
            raise ValueError(
                f"'node_weights' is not supported for {method=}.",
            )
        Xt = np.zeros((X.shape[0], tree.max_depth), dtype=DTYPE)
        paths = tree_estimator.decision_path(X).toarray().astype(bool)
        node_idx = np.arange(tree.node_count)

        for i, path in enumerate(paths):
            is_right_child = tree.children_right[path][:-1] == node_idx[path][1:]
            Xt[i, :len(is_right_child)] = is_right_child

    elif method == "all_nodes":
        node_size = tree.weighted_n_node_samples[mask]

        Xt = X[:, tree.feature[mask]] > tree.threshold[mask]
        Xt = Xt.astype(DTYPE)

        if node_weights is not None:
            Xt *= weights

        Xt = Xt[:, node_size <= max_node_size]
    else:
        raise ValueError

    if Xt.shape[1] == 0 and warn_no_output:
        warnings.warn(
            f"Tree embedder: no nodes were selected, so no features produced."
            f" {tree.max_depth=}. Consider increasing"
            f" max_node_size={original_max_node_size} or {max_pvalue=}."
        )
    return Xt


class BaseTreeEmbedder(
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
):
    _parameter_constraints = {
        "estimator": [BaseEstimator],
        **_COMMON_PARAMS,
    }

    def __init__(
        self,
        estimator,
        method="path",
        node_weights=None,
        max_pvalue=1.0,
        max_node_size=1.0,
    ):
        self.estimator = estimator
        self.method = method
        self.node_weights = node_weights
        self.max_pvalue = max_pvalue
        self.max_node_size = max_node_size

    @_fit_context(
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    @property
    def n_features_in_(self):
        return self.estimator_.n_features_in_


class TreeEmbedder(BaseTreeEmbedder):
    _parameter_constraints = {
        **BaseTreeEmbedder._parameter_constraints,
        "estimator": [BaseDecisionTree],
    }

    def transform(self, X):
        check_is_fitted(self)
        return embed_with_tree(
            tree_estimator=self.estimator_,
            X=X,
            method=self.method,
            node_weights=self.node_weights,
            max_pvalue=self.max_pvalue,
            max_node_size=self.max_node_size,
        )


class ForestEmbedder(BaseTreeEmbedder):
    _parameter_constraints = {
        **BaseTreeEmbedder._parameter_constraints,
        "estimator": [BaseForest],
    }

    def transform(self, X):
        check_is_fitted(self)
        embeddings_iter = (
            embed_with_tree(
                tree,
                X,
                method=self.method,
                node_weights=self.node_weights,
                max_pvalue=self.max_pvalue,
                max_node_size=self.max_node_size,
                warn_no_output=False,
            )
            for tree in self.estimator_.estimators_
        )
        Xt = _hstack(np.fromiter(embeddings_iter, dtype='object'))

        if Xt.shape[1] == 0:
            warnings.warn(
                f"Tree embedder: no nodes were selected, so no features produced."
                f" Consider increasing {self.max_node_size=} or {self.max_pvalue=}."
            )
        return Xt
