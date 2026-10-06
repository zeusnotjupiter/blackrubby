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
