"""El modelo original servido por el tablero (``artifacts/original_servido.joblib``) y lo que publica
``data/prediccion/`` para él."""
import numpy as np
import pytest

from src import datos
from src.data import HOLGURAS, ejemplos_original, original_servido, predecir_original, rangos_original


def test_ejemplos_no_publican_identificadores():
    e = ejemplos_original()
    assert not ({"oc_id", "posicion"} | set(datos.ENTIDADES)) & set(e.columns)
    assert len(e) == 300
    assert e.y_tarde.value_counts().to_dict() == {0: 150, 1: 150}


def test_servido_reproduce_la_2x2():
    _, ficha = original_servido()
    assert 0 < ficha["umbral"] < 1
    assert ficha["reproduccion_2x2"]["misma_decision"] > 0.99


def test_rangos_cubren_los_campos_del_formulario():
    r = rangos_original()
    assert (r.p01 <= r.mediana).all() and (r.mediana <= r.p99).all()
    assert set(r.index) <= set(ejemplos_original().columns)


def test_las_holguras_publicadas_son_plazo_menos_plazo_real():
    # predecir_original las recalcula con esa definición; si no coincidiera, cambiaría la predicción sin tocar nada
    e = ejemplos_original()
    for k in HOLGURAS:
        d = e[f"{k}_holgura_med"] - (e.plazo_habil_al_liberar - e[f"{k}_lt_med"])
        assert np.nanmax(np.abs(d)) < 1e-6


@pytest.mark.parametrize("i", [0, 149, 299])
def test_predecir_da_probabilidad_y_contribuciones(i):
    p, u, p_ln, c = predecir_original(ejemplos_original().iloc[[i]].copy())
    assert 0 <= p <= 1 and 0 < u < 1 and 0 <= p_ln <= 1
    assert c.contribucion.abs().is_monotonic_decreasing
    assert c.variable.is_unique


def test_mas_plazo_baja_el_riesgo():
    f = ejemplos_original().iloc[[0]].copy()
    corto, largo = f.assign(plazo_habil_al_liberar=1.0), f.assign(plazo_habil_al_liberar=60.0)
    assert predecir_original(largo)[2] < predecir_original(corto)[2]


def test_el_riesgo_ordena_las_clases():
    e = ejemplos_original()
    p = np.array([predecir_original(e.iloc[[i]].copy())[0] for i in range(0, len(e), 5)])
    y = e.y_tarde.to_numpy()[::5]
    assert p[y == 1].mean() > p[y == 0].mean()
