"""El XGBoost con Platt que sirve el tablero (``src.calibrado``)."""
import numpy as np
import pytest
import xgboost
from scipy import sparse

from src.calibrado import XGBoostPlatt


@pytest.fixture(scope="module")
def ajustado():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(int)
    return XGBoostPlatt({"n_estimators": 20, "max_depth": 3}, platt=(0.8, 0.1)).fit(X, y), X


def test_probabilidad_es_platt_sobre_el_margen(ajustado):
    m, X = ajustado
    esperado = 1 / (1 + np.exp(-(0.8 * m.margen(X) + 0.1)))
    assert np.allclose(m.predict_proba(X)[:, 1], esperado)


def test_shap_suma_el_margen(ajustado):
    # las contribuciones del tablero (pred_contribs) más el valor base dan el margen, antes de Platt
    m, X = ajustado
    c = m.get_booster().predict(xgboost.DMatrix(X.astype(np.float32)), pred_contribs=True)
    assert np.allclose(c.sum(axis=1), m.margen(X), atol=1e-4)


def test_matriz_dispersa_igual_que_densa(ajustado):
    # un cero disperso no puede volverse un vacío: el one-hot llega disperso
    m, X = ajustado
    X = X.copy()
    X[:, 1] = 0.0
    assert np.allclose(m.predict_proba(sparse.csr_matrix(X)), m.predict_proba(X))
