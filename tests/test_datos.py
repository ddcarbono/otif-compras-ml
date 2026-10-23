"""Preprocesamiento de ``src.datos``: transformaciones numéricas y codificación jerárquica por objetivo.

Usa datos sintéticos: el dataset completo no viaja en el repositorio."""
import numpy as np
import pandas as pd
import pytest

from src import datos


def test_log_signo_conserva_signo_y_orden():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = datos.log_signo(x)
    assert y[2] == 0
    assert np.all(np.sign(y) == np.sign(x))
    assert np.all(np.diff(y) > 0)
    assert y[-1] == pytest.approx(np.log1p(1000))


def test_a_objeto_cambia_na_por_nan():
    s = pd.DataFrame({"c": pd.array(["a", None, "b"], dtype="string")})
    o = datos.a_objeto(s)
    assert o.c.dtype == object
    assert o.c.isna().tolist() == [False, True, False]
    assert isinstance(o.c.iloc[1], float)       # np.nan, no pd.NA


def _toy():
    # dos regiones; el proveedor P3 solo tiene una fila
    df = pd.DataFrame({"proveedor": ["P1"] * 4 + ["P2"] * 4 + ["P3"],
                       "pv_region": ["R1"] * 4 + ["R2"] * 5})
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 1, 1])
    return df, y


def test_codificar_objetivo_sigue_la_formula():
    df, y = _toy()
    glob = y.mean()
    r1 = (3 + datos.M * glob) / (4 + datos.M)
    p1 = (3 + datos.M * r1) / (4 + datos.M)
    cod = datos.codificar_objetivo(df, y, df, ["proveedor"])
    assert cod.te_proveedor.iloc[0] == pytest.approx(p1)


def test_valor_nuevo_recibe_la_tasa_de_su_padre():
    df, y = _toy()
    nuevo = pd.DataFrame({"proveedor": ["NUEVO"], "pv_region": ["R2"]})
    cod = datos.codificar_objetivo(df, y, nuevo, ["proveedor", "pv_region"])
    assert cod.te_proveedor.iloc[0] == pytest.approx(cod.te_pv_region.iloc[0])


def test_aplicar_tablas_reproduce_codificar_objetivo():
    df, y = _toy()
    tablas, glob = datos.tablas_objetivo(df, y)
    a = datos.aplicar_tablas(tablas, glob, df, ["proveedor", "pv_region"])
    b = datos.codificar_objetivo(df, y, df, ["proveedor", "pv_region"])
    pd.testing.assert_frame_equal(a, b)


def test_fuera_de_pliegue_no_ve_su_propia_orden():
    # cada orden tiene un proveedor propio: fuera de pliegue, ninguna fila ve su objetivo y todas caen a la región
    n = 10
    df = pd.DataFrame({"proveedor": [f"P{i}" for i in range(n)], "pv_region": ["R"] * n,
                       "oc_id": [f"OC{i}" for i in range(n)]})
    y = pd.Series([1, 0] * (n // 2))
    out = datos.codificar_fuera_de_pliegue(df, y, df.oc_id, ["proveedor"])
    dentro = datos.codificar_objetivo(df, y, df, ["proveedor"])
    # dentro de muestra las tardías quedan arriba de las puntuales; fuera de pliegue eso desaparece
    assert dentro.te_proveedor[y == 1].min() > dentro.te_proveedor[y == 0].max()
    assert out.te_proveedor[y == 1].mean() <= out.te_proveedor[y == 0].mean()


def test_codificador_jerarquico_transform_usa_todo_lo_visto():
    df, y = _toy()
    df["oc_id"] = [f"OC{i}" for i in range(len(df))]
    c = datos.CodificadorJerarquico(columnas=["proveedor"]).fit(df, y)
    esperado = datos.codificar_objetivo(df, y, df, ["proveedor"]).to_numpy()
    np.testing.assert_allclose(c.transform(df), esperado)
    assert c.get_feature_names_out().tolist() == ["te_proveedor"]


def test_tipos_del_diccionario_no_se_pisan():
    assert not set(datos.NUMERICAS) & set(datos.CATEGORICAS)
    assert not set(datos.CATEGORICAS) & set(datos.ENTIDADES)
    assert datos.OBJETIVO not in datos.PREDICTORAS
    assert "oc_id" not in datos.PREDICTORAS


def test_particion():
    df = pd.DataFrame({"particion": ["entrenamiento", "validacion", "prueba", "validacion"]})
    assert len(datos.particion(df, "validacion")) == 2
