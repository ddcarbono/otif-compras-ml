"""Series de tiempo del EDA (``src.tiempo``) sobre datos sintéticos."""
import numpy as np
import pandas as pd

from src import tiempo


def _df():
    rng = np.random.default_rng(0)
    fechas = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, 5000), unit="D")
    return pd.DataFrame({"fecha_pedido": fechas, "y_tarde": rng.integers(0, 2, 5000), "linea": rng.choice(["a", "b"], 5000)})


def test_serie_mensual_cuenta_todo():
    df = _df()
    s = tiempo.serie(df)
    assert s.filas.sum() == len(df) and s.tardias.sum() == df.y_tarde.sum()
    assert s.tasa.between(0, 1).all()
    assert tiempo.serie(df, por="linea").filas.sum() == len(df)


def test_huecos_cuenta_dias():
    h = tiempo.huecos(_df()).set_index("medida").valor
    assert int(h["días con órdenes"]) <= int(h["días del periodo"])


def test_ruido_blanco_es_estacionario():
    s = np.random.default_rng(1).normal(size=200)
    e = tiempo.estacionariedad(s).set_index("prueba")
    assert e.loc["ADF", "p"] < 0.05     # ADF rechaza la raíz unitaria
    assert e.loc["KPSS", "p"] > 0.05    # KPSS no rechaza la estacionariedad


def test_paseo_aleatorio_no_es_estacionario():
    s = np.cumsum(np.random.default_rng(2).normal(size=200))
    # KPSS rechaza la estacionariedad; ADF, con 200 puntos, a veces rechaza la raíz unitaria por azar
    assert tiempo.estacionariedad(s).set_index("prueba").loc["KPSS", "p"] < 0.05


def test_stl_reconstruye_la_serie():
    m = tiempo.serie(_df())
    d = tiempo.descomponer(m)
    np.testing.assert_allclose(d.tendencia + d.estacionalidad + d.residuo, d.observada)
