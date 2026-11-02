import numpy as np
import pytest

from blackrubby import BlackRubbyRegressor


def test_multifeature_fit_predict_score_and_repeatability():
    rng = np.random.default_rng(20261005)
    X = rng.uniform(-1.0, 1.0, size=(512, 5))
    y = (np.sin(2 * X[:, 0] * X[:, 1]) + 0.2 * X[:, 0] - 0.1 * X[:, 1]
         + 0.3 * X[:, 2] ** 2 + 0.1 * X[:, 3] * X[:, 4])

    options = dict(expansion_dim=256, select_dim=64, select_rows=256,
                   sym_population=64, sym_generations=40, sym_time_limit=0.0,
                   random_state=77)
    model = BlackRubbyRegressor(**options).fit(X[:400], y[:400])
    repeated = BlackRubbyRegressor(**options).fit(X[:400], y[:400])

    predictions = model.predict(X[400:])
    assert model.n_features_in_ == 5
    assert predictions.shape == (112,)
    assert np.isfinite(predictions).all()
    assert model.score(X[400:], y[400:]) > 0.90
    np.testing.assert_allclose(predictions, repeated.predict(X[400:]), rtol=0, atol=1e-10)


def test_validates_feature_count_and_parameters():
    X = np.arange(40.0).reshape(20, 2)
    y = X[:, 0] - X[:, 1]
    model = BlackRubbyRegressor(expansion_dim=64, use_symbolic=False).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.ones((3, 3)))
    model.set_params(alpha=0.1)
    assert model.is_fitted_ is False
    assert model.get_params()["alpha"] == 0.1


def test_accepts_array_like_inputs():
    X = np.arange(40.0).reshape(20, 2)
    y = X[:, 0] - X[:, 1]
    expected = BlackRubbyRegressor(expansion_dim=64, use_symbolic=False).fit(X, y).predict(X)
    from_lists = BlackRubbyRegressor(expansion_dim=64, use_symbolic=False).fit(
        X.tolist(), y.tolist()).predict([[0.0, 1.0]])
    np.testing.assert_allclose(from_lists, expected[:1])

    pd = pytest.importorskip("pandas")
    from_pandas = BlackRubbyRegressor(expansion_dim=64, use_symbolic=False).fit(
        pd.DataFrame(X, columns=["a", "b"]), pd.Series(y)).predict(pd.DataFrame(X))
    np.testing.assert_allclose(from_pandas, expected)


def test_pickle_round_trip_reproduces_predictions():
    import pickle

    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, size=(128, 3))
    y = np.sin(X[:, 0] * X[:, 1]) + 0.2 * X[:, 2]
    options = dict(expansion_dim=128, select_dim=32, select_rows=128,
                   sym_population=32, sym_generations=10, random_state=5)
    model = BlackRubbyRegressor(**options).fit(X, y)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=0, atol=0)
    assert restored.champion_formula_ == model.champion_formula_
    assert restored.n_features_in_ == 3

    unfitted = pickle.loads(pickle.dumps(BlackRubbyRegressor(**options)))
    assert unfitted.is_fitted_ is False
    with pytest.raises(RuntimeError):
        unfitted.predict(X)


def test_sklearn_interop():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(120, 3))
    y = np.sin(X[:, 0] * X[:, 1]) + 0.2 * X[:, 2]
    fast = dict(expansion_dim=64, select_dim=32, select_rows=120,
                sym_population=32, sym_generations=5)

    model = BlackRubbyRegressor(**fast)
    cloned = clone(model)
    assert cloned.get_params()["expansion_dim"] == 64

    scores = cross_val_score(BlackRubbyRegressor(**fast), X, y, cv=2)
    assert np.isfinite(scores).all()

    pipe = Pipeline([("model", BlackRubbyRegressor(**fast))]).fit(X, y)
    assert np.isfinite(pipe.score(X, y))

    grid = GridSearchCV(BlackRubbyRegressor(**fast), {"alpha": [0.0, 0.1]}, cv=2).fit(X, y)
    assert grid.best_params_["alpha"] in (0.0, 0.1)
