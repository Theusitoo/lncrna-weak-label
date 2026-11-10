from numbers import Real
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from warnings import warn

import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    check_is_fitted,
    _fit_context,
)
import sklearn.utils
from sklearn.utils._param_validation import (
    Interval, StrOptions, HasMethods, Hidden,
)
from imblearn.base import BaseSampler


def _standardize_proba(proba):
    if isinstance(proba, list):
        return proba
    if isinstance(proba, np.ndarray):
        if proba.ndim == 2:
            return [proba]
        if proba.ndim == 3:
            return list(proba.transpose(2, 0, 1))

    shape = getattr(proba, "shape", None) or f"{len(proba)=}"
    raise ValueError(
        "Expected proba to be a list or numpy array of shape"
        " (n_samples, n_classes) or (n_samples, n_classes, n_outputs)."
        f" Got {type(proba)=}. Shape: {shape}."
    )


def _get_classes(estimator):
    if hasattr(estimator, "classes_"):
        return estimator.classes_
    if estimator.n_outputs_ == 1:
        return np.array([0, 1])
    return [np.array([0, 1])] * estimator.n_outputs_


class WeakLabelImputer(BaseSampler, MetaEstimatorMixin):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "predict"]),
        ],
        "threshold": [Interval(Real, 0, 1, closed="left")],
        "use_oob_proba": ["boolean"],
        "weight_proba": ["boolean"],
        "verbose": ["verbose"],
        "sampling_strategy": [StrOptions({"auto"})],
    }

    def __init__(
        self,
        estimator,
        threshold=0.8,
        use_oob_proba=False,
        weight_proba=False,
        verbose=False,
        sampling_strategy="auto",
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.use_oob_proba = use_oob_proba
        self.weight_proba = weight_proba
        self.verbose = verbose
        self.sampling_strategy = sampling_strategy

    def _weight_proba(self, proba, y, classes):
        if not self.weight_proba:
            return proba

        new_probas = []
        for proba_col, y_col, class_names in zip(proba, y.T, classes):
            counts_idx, counts = np.unique(y_col, return_counts=True)
            counts_order = [np.where(counts_idx == c)[0][0] for c in class_names]
            counts = counts[counts_order]
            proba_col *= y.shape[0] / counts
            proba_col /= proba_col.sum(axis=1, keepdims=True)
            new_probas.append(proba_col)

        return new_probas

    def _apply_threshold(self, proba, threshold):
        return np.hstack(
            [np.max(p, axis=1, keepdims=True) < threshold for p in proba]
        )

    def _proba_to_pred(self, estimator, proba, n_samples):
        classes = _get_classes(estimator)

        if estimator.n_outputs_ == 1:
            return classes.take(np.argmax(proba[0], axis=1), axis=0).reshape(-1, 1)

        proba = np.dstack(proba).transpose(0, 2, 1)

        class_type = classes[0].dtype
        predictions = np.zeros((n_samples, estimator.n_outputs_), dtype=class_type)

        for k in range(estimator.n_outputs_):
            predictions[:, k] = classes[k].take(
                np.argmax(proba[:, k], axis=1), axis=0
            )

        return predictions

    def _fit_predict_proba(self, X, y, **params):
        estimator = clone(self.estimator).fit(X, y, **params)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if is_classifier(estimator):
            if self.use_oob_proba:
                proba = estimator.oob_decision_function_
            else:
                proba = estimator.predict_proba(X)
        else:
            if self.use_oob_proba:
                proba = estimator.oob_prediction_
            else:
                proba = estimator.predict(X)

            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)

            proba = np.dstack([1 - proba, proba]).transpose(0, 2, 1)

        proba = _standardize_proba(proba)

        return estimator, proba

    @_fit_context(
        prefer_skip_nested_validation=False,
    )
    def _fit_resample(self, X, y, **params):
        estimator, proba = self._fit_predict_proba(X, y, **params)
        proba = self._weight_proba(proba, y, _get_classes(estimator))

        kept = self._apply_threshold(proba, self.threshold)

        y_pred = self._proba_to_pred(estimator, proba, y.shape[0])

        y = y.reshape(y_pred.shape)
        kept = kept.reshape(y_pred.shape)

        if self.verbose:
            n_changed = (y_pred[~kept] != y[~kept]).sum()
            self.print_message(
                f"Changing {n_changed}/{y.size} ({n_changed / y.size:.2%}) labels."
            )

        y_pred[kept] = y[kept]

        return X, y_pred

    def fit_resample(self, X, y, **params):
        return self._fit_resample(X, y, **params)

    def print_message(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}] " + message)


class PositiveUnlabeledImputer(WeakLabelImputer):
    def _fit_resample(self, X, y):
        Xt, yt = super().fit_resample(X, y)
        final_y = (y.astype(bool) | yt.astype(bool)).astype(int)

        if self.verbose:
            n_changed = (final_y != y).sum()
            n_pos = y.sum()
            self.print_message(
                f"Imputted {n_changed}/{n_pos:.0f}"
                f" ({n_changed / n_pos:.2%}) positive labels."
            )

        return Xt, final_y


class SCARImputer(WeakLabelImputer):
    _parameter_constraints = {
        **WeakLabelImputer._parameter_constraints,
        "label_freq_percentile": [Interval(Real, 0, 1, closed="neither")],
        "last_level": [Hidden(None), Hidden(WeakLabelImputer)],
    }

    def __init__(
        self,
        estimator,
        *,
        label_freq_percentile=0.5,
        threshold=0.5,
        verbose=False,
        use_oob_proba=True,
        last_level: Self | None = None,
    ):
        self.last_level = last_level
        self.label_freq_percentile = label_freq_percentile

        super().__init__(
            estimator=estimator,
            threshold=threshold,
            verbose=verbose,
            weight_proba=True,
            sampling_strategy="auto",
            use_oob_proba=use_oob_proba,
        )

    def _estimate_label_frequencies(self, y_original, proba, X=None, y=None):
        label_frequency_estimates = []

        for y_col, proba_col in zip(y_original.T, proba):
            proba_col = proba_col[:, 1]
            assert proba_col.shape == y_col.shape

            positives_proba = proba_col[y_col.astype(bool)]
            if len(positives_proba) == 0:
                # No positive samples in this fold for this label:
                # fall back to a small frequency estimate (1 / n_samples).
                freq_estimate = 1.0 / (len(y_col) + 1)
            else:
                freq_estimate = np.percentile(
                    positives_proba,
                    100 * self.label_freq_percentile,
                )
            label_frequency_estimates.append(freq_estimate)

        self.print_message(
            f"Estimated label frequencies (c) for each output:"
            f" [{' '.join(f'{c:.2f}' for c in label_frequency_estimates)}]"
        )

        return np.array(label_frequency_estimates)

    def _weight_proba(self, proba, y, classes):
        if not self.weight_proba:
            return proba

        new_probas = []
        for proba_col, label_freq in zip(
            proba,
            self.label_frequency_estimates_[self.enable_imputation_]
        ):
            proba_col = proba_col.copy()
            proba_col[:, 1] = (proba_col[:, 1] / label_freq).clip(0, 1)
            proba_col[:, 0] = 1 - proba_col[:, 1]
            new_probas.append(proba_col)

        return new_probas

    def _apply_threshold(self, proba, threshold):
        keep = super()._apply_threshold(proba, threshold)
        active_original_y = self.original_y_[:, self.enable_imputation_]
        keep[active_original_y.astype(bool)] = True
        return keep

    def _fit_predict_proba(self, X, y, **params):
        if self.last_level is None:
            self.enable_imputation_ = np.ones(y.shape[1], dtype=bool)
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = y
        else:
            self.enable_imputation_ = self.last_level.next_enable_imputation_.copy()
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = self.last_level.original_y_

            if not self.enable_imputation_.any():
                self.print_message("No labels to impute.")
                return None, y

        active_y = y[:, self.enable_imputation_]
        estimator, proba = super()._fit_predict_proba(X, active_y, **params)

        self.label_frequency_estimates_ = (
            self._estimate_label_frequencies(
                self.original_y_[:, self.enable_imputation_],
                proba,
                X=X,
                y=active_y,
            )
        )
        return estimator, proba

    def _fit_resample(self, X, y, **params):
        estimator, proba = self._fit_predict_proba(X, y, **params)

        if estimator is None:
            return X, y

        active_y = y[:, self.enable_imputation_]
        active_original_y = self.original_y_[:, self.enable_imputation_]

        proba = self._weight_proba(
            proba, active_original_y, _get_classes(estimator),
        )

        kept = self._apply_threshold(proba, self.threshold)

        y_pred = self._proba_to_pred(estimator, proba, active_y.shape[0])

        active_y = active_y.reshape(y_pred.shape)
        kept = kept.reshape(y_pred.shape)

        if self.verbose:
            n_changed = (y_pred[~kept] != active_y[~kept]).sum()
            self.print_message(
                f"Imputting {n_changed}/{y.size}"
                f" ({n_changed / y.size:.2%}) labels relative to last layer."
            )

        y_pred[kept] = active_original_y.reshape(y_pred.shape)[kept]
        y_pred_full = y.copy()
        y_pred_full[:, self.enable_imputation_] = y_pred

        return X, y_pred_full


class LabelComplementImputer(SCARImputer):
    def __init__(
        self,
        estimator,
        *,
        label_freq_percentile=0.5,
        threshold=0.5,
        weight_proba=False,
        verbose=False,
        use_oob_proba=True,
        last_level: Self | None = None,
    ):
        self.weight_proba = weight_proba

        super().__init__(
            estimator=estimator,
            threshold=threshold,
            verbose=verbose,
            last_level=last_level,
            label_freq_percentile=label_freq_percentile,
            use_oob_proba=use_oob_proba,
        )

    def _fit_predict_proba(self, X, y, **params):
        if self.last_level is None:
            return super()._fit_predict_proba(X, y, **params)

        self.enable_imputation_ = (
            self.last_level.next_enable_imputation_.copy()
        )
        self.next_enable_imputation_ = self.enable_imputation_.copy()

        self.label_frequency_estimates_ = (
            self.last_level.label_frequency_estimates_.copy()
        )
        self.original_y_ = self.last_level.original_y_

        if not self.enable_imputation_.any():
            self.print_message("No labels to impute.")
            return None, y

        active_y = y[:, self.enable_imputation_]

        return WeakLabelImputer._fit_predict_proba(self, X, active_y, **params)

    def _apply_threshold(self, proba, threshold):
        keep_masks = []
        n_samples = proba[0].shape[0]

        for i, proba_col, y_col, label_freq in zip(
            np.nonzero(self.enable_imputation_)[0],
            proba,
            self.original_y_.T[self.enable_imputation_],
            self.label_frequency_estimates_[self.enable_imputation_],
        ):
            keep_mask = np.ones_like(y_col, dtype=bool)
            proba_col = proba_col[:, 1]

            max_labels = int(min(np.sum(y_col) / label_freq, n_samples))

            if max_labels == n_samples:
                warn(
                    "Expected number of labels is too large:"
                    f" {np.sum(y_col) / label_freq=} > {n_samples=}"
                )

            change_idx = np.argsort(proba_col + 2 * y_col)[::-1][:max_labels]
            keep_mask[change_idx] = False

            keep_mask |= proba_col < threshold

            n_labeled = ((~keep_mask) | y_col.astype(bool)).sum()

            if n_labeled >= max_labels:
                self.next_enable_imputation_[i] = False
                self.print_message(f"Stopped imputation for output {i}.")
            if n_labeled > max_labels:
                warn(
                    f"Imputed more labels than expected: {n_labeled=} > {max_labels=}"
                )

            keep_mask |= y_col.astype(bool)

            keep_masks.append(keep_mask)

        return np.vstack(keep_masks).T
