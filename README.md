# BlackRubby Regressor

BlackRubby is a numeric regression estimator that fuses two complementary learners:

- **Rubby** — a symbolic search that evolves closed-form candidate expressions (e.g. `sqrt(x0^3 / (x1 + x2))`) on a 3×3×3 cube-walk genome.
- **Black** — a random-Fourier-feature ridge regression that fits a smooth kernel model over the inputs.

The final model is a single ridge over `[Fourier features | standardized inputs | symbolic columns]`, where the bandwidth, ridge strength, and whether to trust the symbolic columns are all chosen by exact leave-one-out cross-validation. It handles multiple numeric input features and one numeric target, with no Python-side dependencies beyond NumPy.

```python
import numpy as np
from blackrubby import BlackRubbyRegressor

rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(512, 3))
y = np.sin(X[:, 0] * X[:, 1]) + 0.2 * X[:, 2]

model = BlackRubbyRegressor(expansion_dim=256, random_state=7)
model.fit(X[:400], y[:400])
print("held-out R2:", model.score(X[400:], y[400:]))
print("champion:", model.champion_formula_)
```

## Install

```bash
python -m pip install blackrubby-regressor
```

Pre-built wheels are published for CPython 3.10–3.13 on Linux (x86_64), Windows (x64), and macOS (Intel and Apple Silicon). On a supported platform this installs in seconds with no compiler.

Building from source requires only a C++17 compiler and CMake. Eigen 3.4.0 is vendored under `vendor/eigen`, so **no network access is needed**; set `EIGEN3_INCLUDE_DIR` to use a different Eigen installation.

## How it works

Given training pairs `(x_i, y_i)` with `x_i ∈ R^d`, BlackRubby fits a linear model in a lifted feature space:

```
ŷ(x) = wᵀ Φ(x) + b
```

The feature map `Φ(x)` has three blocks:

1. **Random Fourier features.** `D` cos/sin pairs approximating a stationary kernel:
   `Φ_fourier(x) = (1/√D) [cos(W x̂ / γ), sin(W x̂ / γ)]`, where `x̂` is the standardized input, `W ∈ R^{D×d}` has standard-normal entries, and `γ` is the bandwidth. This is the classic Rahimi–Recht approximation — as `D → ∞`, the inner product `Φ(x)ᵀΦ(x')` converges to the Gaussian kernel `k(x, x') = exp(-‖x̂ - x̂'‖² / 2γ²)`.

2. **Standardized inputs.** The raw `(x - μ) / σ` block, which lets the model represent exactly-linear components without spending Fourier capacity on them.

3. **Symbolic columns (Rubby).** Up to `sym_top_k` expressions found by the search, each standardized to zero mean and unit variance. These give the ridge a direct handle on non-smooth or strongly multiplicative structure (ratios, square roots, powers) that Fourier features approximate only slowly.

The ridge weights minimize the regularized squared error:

```
w* = argmin_w  Σ_i (y_i - wᵀΦ(x_i) - b)² + α ‖w‖²
```

### Hyperparameters by exact leave-one-out CV

`bandwidth γ`, ridge strength `α`, and the symbolic on/off decision are chosen on a `select_rows`-row subset using the **exact leave-one-out** formula for ridge regression. One eigendecomposition of the centered Gram matrix gives the LOO residual for every `α` in the grid simultaneously:

```
e_i^LOO = (y_i - ŷ_i) / (1 - h_i),   h = U diag(λ_j / (λ_j + nα)) Uᵀ
```

so the model only leans on the symbolic columns when they demonstrably reduce out-of-sample error. A 2-fold holdout at full feature width re-checks the symbolic decision, because the reduced-width selection screen can mis-rank those columns.

### The Rubby symbolic search

Candidate expressions are encoded as a walk on a 3×3×3 cube whose 27 cells each hold one operation or operand (`const`, `x_k`, `+`, `-`, `*`, `/`, `log`, `exp`, `sqrt`, `abs`, `x²`, `x³`). Starting from the corner cell, the genome's path (`sym_turns` moves, each ±x/±y/±z with wraparound) visits cells in order, and the visited nodes form a stack program evaluated column-wise over the data. Each candidate gets a least-squares scale and offset (`a·f(x) + b`), so the search only has to find the right *shape*.

A generation consists of tournament selection (with a path-distance diversity check), uniform crossover over cube cells, point mutation, occasional MCTS refinement of the walk itself (UCT over path prefixes), and a periodic Levenberg–Marquardt polish of the constants on the active path. Fitness is normalized mean squared error, `NMSE = MSE / Var(y)`. The best distinct expressions form a hall of fame that feeds the ridge.

The search runs on a strided `sym_rows`-row subsample for `sym_generations` generations and is fully seeded by `random_state`.

### Final solve

The full model is solved in primal form over up to `solve_rows` rows (auto-capped at `max(50000, 20 × n_features)`), accumulating the centered Gram matrix `G = ΦᵀΦ` in batches and applying a Cholesky solve with an LDLT fallback.

## Reproducibility

With the default `sym_time_limit=0`, the symbolic search is generation-bounded and **fully deterministic**: the same `random_state`, data, and parameters produce a bit-identical model on every run, regardless of thread count or machine load. Setting `sym_time_limit > 0` switches to a wall-clock cap, which can stop at different generations on different runs and therefore trades reproducibility for a hard time budget.

## API

```python
BlackRubbyRegressor(
    expansion_dim=1024,    # Fourier cos/sin pairs (D)
    alpha=0.0,             # ridge strength; 0 = auto by LOO-CV
    bandwidth=0.0,         # Fourier bandwidth γ; 0 = auto by LOO-CV
    random_state=42,       # seed for projections and symbolic search
    use_symbolic=True,     # run the Rubby search
    sym_population=128,    # GA population size
    sym_generations=100,   # GA generations (when time limit is 0)
    sym_rows=1024,         # rows used by the symbolic search
    sym_top_k=6,           # symbolic columns offered to the ridge
    sym_time_limit=0.0,    # seconds; 0 = generation-bounded (reproducible)
    select_rows=2048,      # rows used for hyperparameter selection
    select_dim=128,        # Fourier pairs used during selection
    batch_size=2048,       # rows per block in feature assembly
    solve_rows=-1,         # rows in the final solve; -1 = auto cap, 0 = all
    max_frequency=54.598,  # largest 1D frequency (standardized units); 1D inputs only
)
```

Methods: `fit(X, y)`, `predict(X)`, `score(X, y)` (R²), `get_params()`, `set_params(**params)`.

After fitting, the following attributes are available:

| Attribute | Meaning |
|---|---|
| `n_features_in_` | number of input columns seen during `fit` |
| `fit_seconds_` | total wall-clock fit time |
| `symbolic_seconds_` | time spent in the Rubby search |
| `chosen_bandwidth_` | selected Fourier bandwidth γ |
| `chosen_alpha_` | selected ridge strength α |
| `uses_symbolic_` | whether the LOO/holdout checks kept the symbolic columns |
| `symbolic_columns_` | number of symbolic columns in the final model |
| `champion_formula_` | best expression found, in infix (e.g. `0.9228*exp(x0*x1) - 0.0061`) |

## Usage notes

- `X` must be a finite 1D or 2D numeric array; `y` must be one finite numeric target per row. Multi-output regression, categorical encoding, and missing-value imputation are not built in — encode and impute before calling `fit`.
- Fit on training data only, tune on validation data, and report on an untouched test set. The internal LOO-CV protects hyperparameter selection, but it is not a substitute for a held-out evaluation.
- For 1D inputs the Fourier projection is a deterministic log-spaced frequency grid up to `max_frequency`, which suits signal-like data; for `d ≥ 2` the projection is a seeded random Gaussian matrix.
- `score(X, y)` returns the coefficient of determination R² (1.0 is perfect, 0.0 is no better than predicting the mean).

## Build from source

```bash
git clone --recurse-submodules https://github.com/zeusnotjupiter/BlackRubby.git
cd BlackRubby
python -m pip install build
python -m build
```

(If you cloned without `--recurse-submodules`, run `git submodule update --init` first — the Eigen headers under `vendor/eigen` are required.)

For an editable development build:

```bash
python -m pip install scikit-build-core pybind11 cmake numpy
python -m pip install -e . --no-build-isolation
```

Run the tests with `python -m pip install pytest` then `python -m pytest tests`.

## Release and publishing

Wheels for all supported platforms are built by the `wheels.yml` workflow (cibuildwheel) on version tags and can be test-built at any time via manual dispatch. Publishing to PyPI is a separate, explicitly gated step: it requires a PyPI project named `blackrubby-regressor`, this repository configured as a trusted publisher, and a manual workflow dispatch with `publish_to_pypi` enabled. No credentials are stored in the repository.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
