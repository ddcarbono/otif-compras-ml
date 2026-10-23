"""El modelo servido por el tablero (``artifacts/servido.joblib``) y lo que publica ``data/prediccion/``."""
import json

import numpy as np
import pytest

from src import datos
from src.data import ejemplos, predecir, rangos, servido


def test_ejemplos_no_publican_identificadores():
    e = ejemplos()
    assert not ({"oc_id", "posicion"} | set(datos.ENTIDADES)) & set(e.columns)
    assert len(e) == 300
    assert e.y_tarde.value_counts().to_dict() == {0: 150, 1: 150}


def test_servido_trae_columnas_y_umbral():
    s, ficha = servido()
    assert 0 < ficha["umbral"] < 1
    assert s["num_cols"] == datos.NUMERICAS and s["cat_cols"] == datos.CATEGORICAS
    assert all(c.startswith("te_") for c in s["te_cols"])


def test_rangos_cubren_los_campos_del_formulario():
    r = rangos()
    for c, f in r.iterrows():
        if f.tipo == "categórica":
            assert json.loads(f.categorias)
        else:
            assert f.p01 <= f.mediana <= f.p99


@pytest.mark.parametrize("i", [0, 149, 299])
def test_predecir_da_probabilidad_y_contribuciones(i):
    p, u, c = predecir(ejemplos().iloc[[i]].copy())
    assert 0 <= p <= 1 and 0 < u < 1
    assert c.contribucion.abs().is_monotonic_decreasing
    assert c.variable.is_unique


def test_contribuciones_de_la_logistica_suman_el_logit():
    s, _ = servido()
    if not hasattr(s["modelo"], "coef_"):
        pytest.skip("el modelo servido no es lineal")
    p, _, c = predecir(ejemplos().iloc[[0]].copy())
    logit = np.log(p / (1 - p))
    assert c.contribucion.sum() + s["modelo"].intercept_[0] == pytest.approx(logit, abs=1e-6)


def test_el_riesgo_ordena_las_clases():
    # sobre los 300 ejemplos, las posiciones tardías reciben en promedio más riesgo que las puntuales
    e = ejemplos()
    p = np.array([predecir(e.iloc[[i]].copy())[0] for i in range(0, len(e), 5)])
    y = e.y_tarde.to_numpy()[::5]
    assert p[y == 1].mean() > p[y == 0].mean()
