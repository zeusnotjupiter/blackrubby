"""BlackRubby symbolic-plus-kernel regression."""

import numpy as np

from ._core import BlackRubbyRegressor as _NativeRegressor

__version__ = "0.1.3"
__all__ = ["BlackRubbyRegressor", "__version__"]

_PARAMS = (
    "expansion_dim", "alpha", "bandwidth", "random_state", "use_symbolic",
    "sym_population", "sym_generations", "sym_rows", "sym_top_k",
    "sym_time_limit", "select_rows", "select_dim", "batch_size",
    "solve_rows", "max_frequency",
)

try:  # scikit-learn is optional; the estimator works identically without it.
    from sklearn.base import BaseEstimator, RegressorMixin

    _BASES = (BaseEstimator, RegressorMixin)
except ImportError:
    _BASES = ()


class BlackRubbyRegressor(*_BASES):
    """Symbolic-plus-kernel regression estimator.

    Accepts any array-like input (NumPy arrays, lists, pandas DataFrames and
    Series), is picklable once fitted, and follows the scikit-learn estimator
    API when scikit-learn is installed. See the project README for parameter
    details.
    """

    def __init__(self, expansion_dim=1024, alpha=0.0, bandwidth=0.0, random_state=42,
                 use_symbolic=True, sym_population=128, sym_generations=100, sym_rows=1024,
                 sym_top_k=6, sym_time_limit=0.0, select_rows=2048, select_dim=128,
                 batch_size=2048, solve_rows=-1, max_frequency=54.5981500331):
        self.expansion_dim = expansion_dim
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.use_symbolic = use_symbolic
        self.sym_population = sym_population
        self.sym_generations = sym_generations
        self.sym_rows = sym_rows
        self.sym_top_k = sym_top_k
        self.sym_time_limit = sym_time_limit
        self.select_rows = select_rows
        self.select_dim = select_dim
        self.batch_size = batch_size
        self.solve_rows = solve_rows
        self.max_frequency = max_frequency
        self._build_impl()

    def _build_impl(self):
        self._impl = _NativeRegressor(**{name: getattr(self, name) for name in _PARAMS})
        self._sync()

    def _sync(self):
        impl = self._impl
        self.is_fitted_ = impl.is_fitted_
        self.n_features_in_ = impl.n_features_in_
        self.fit_seconds_ = impl.fit_seconds_
        self.symbolic_seconds_ = impl.symbolic_seconds_
        self.chosen_bandwidth_ = impl.chosen_bandwidth_
        self.chosen_alpha_ = impl.chosen_alpha_
        self.uses_symbolic_ = impl.uses_symbolic_
        self.symbolic_columns_ = impl.symbolic_columns_
        self.champion_formula_ = impl.champion_formula_

    def fit(self, X, y):
        self._impl.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._sync()
        return self

    def predict(self, X):
        return self._impl.predict(np.asarray(X, dtype=np.float64))

    def score(self, X, y):
        return self._impl.score(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))

    def set_params(self, **params):
        if _BASES:
            super().set_params(**params)
        else:
            for key, value in params.items():
                if key not in _PARAMS:
                    raise ValueError(f"unknown parameter: {key}")
                setattr(self, key, value)
        self._build_impl()
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        impl = state.pop("_impl")
        state["_impl_state"] = impl.get_state() if impl.is_fitted_ else None
        return state

    def __setstate__(self, state):
        impl_state = state.pop("_impl_state")
        self.__dict__.update(state)
        self._build_impl()
        if impl_state is not None:
            self._impl.set_state(impl_state)
            self._sync()

    if not _BASES:

        def get_params(self, deep=True):
            return {name: getattr(self, name) for name in _PARAMS}
