# BlackRubby Regressor

BlackRubby is an experimental numeric regression estimator. It combines a Rubby symbolic-expression search with a Black random-Fourier-feature ridge model. It supports multiple numeric input features and one numeric target.

## Install

After publication:

```bash
python -m pip install blackrubby-regressor
```

The project is not published yet. The current local Windows/Python 3.12 wheel can be installed with:

```bash
python -m pip install dist/blackrubby_regressor-0.1.0-cp312-cp312-win_amd64.whl
```

Release wheels bundle the compiled C++ extension. Building from source requires a C++17 compiler, CMake, and network access to fetch Eigen 3.4.0 during the build. Set `EIGEN3_INCLUDE_DIR` to an existing Eigen include directory to skip the download.

## Example

```python
import numpy as np
from blackrubby import BlackRubbyRegressor

rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(512, 3))
y = np.sin(X[:, 0] * X[:, 1]) + 0.2 * X[:, 2]

model = BlackRubbyRegressor(expansion_dim=256, random_state=7)
model.fit(X[:400], y[:400])
predictions = model.predict(X[400:])
print("held-out R2:", model.score(X[400:], y[400:]))
```

`X` must be a finite 1D or 2D numeric array. `y` must be one finite numeric target per row; multi-output regression, categorical encoding, and missing-value imputation are not included. Fit on training data only, tune using validation data, and report performance on an untouched test set. The symbolic search is seeded by `random_state`; a positive `sym_time_limit` is a wall-clock cap and can stop at different generations on different runs. Set `sym_time_limit=0` for repeatable generation-bounded search, controlled by `sym_generations`.

Common estimator methods are `fit(X, y)`, `predict(X)`, `score(X, y)`, `get_params()`, and `set_params(**params)`. After fitting, `n_features_in_`, `fit_seconds_`, `chosen_bandwidth_`, `chosen_alpha_`, `uses_symbolic_`, and `champion_formula_` expose fit information.

## Build locally

From this directory:

```bash
python -m pip install build
python -m build
```

For a local editable build, install `scikit-build-core`, `pybind11`, and `cmake`, then run `python -m pip install -e .`.

## Release status

This project is experimental. The Python API and a Windows/Python 3.12 wheel have been tested, but the project is not published. Before a public release, the owner must choose and add a license, verify the project name is available on PyPI, configure the repository as a trusted PyPI publisher, build and test wheels on all supported platforms, and explicitly dispatch the publish workflow. No credentials belong in source files.
