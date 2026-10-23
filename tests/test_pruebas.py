"""DeLong contra pROC, delta de Cliff e intervalos BCa (``src.pruebas``)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import pruebas

DATOS = Path(__file__).parent / "datos"


@pytest.mark.parametrize("caso", range(1, 7))
def test_delong_coincide_con_proc(caso):
    # los casos y lo que da pROC::roc.test los escribe src/validar_delong.R (con empates en 4 de los 6)
    d = pd.read_csv(DATOS / "delong_casos.csv").query("caso == @caso")
    e = pd.read_csv(DATOS / "delong_proc.csv").query("caso == @caso").iloc[0]
    o = pruebas.delong(d.y, d.s1, d.s2)
    assert o["AUC_1"] == pytest.approx(e.AUC_1, abs=1e-9)
    assert o["AUC_2"] == pytest.approx(e.AUC_2, abs=1e-9)
    assert o["z"] == pytest.approx(e.z, abs=1e-7)
    assert o["p"] == pytest.approx(e.p, abs=1e-7)


def test_auc_es_el_de_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(y + rng.normal(size=500), 1)          # con empates
    assert pruebas.auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_delong_simetrico_y_nulo_con_el_mismo_puntaje():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    a, b = y + rng.normal(size=300), y + rng.normal(size=300)
    ab, ba = pruebas.delong(y, a, b), pruebas.delong(y, b, a)
    assert ab["z"] == pytest.approx(-ba["z"]) and ab["p"] == pytest.approx(ba["p"])
    assert pruebas.delong(y, a, a)["p"] == 1.0


def test_cliff():
    assert pruebas.cliff([4, 5, 6], [1, 2, 3]) == (1.0, "grande")
    assert pruebas.cliff([1, 2, 3], [1, 2, 3])[0] == 0
    # 10 pares a favor y 3 en contra de 16: δ = 7/16 = 0,4375
    assert pruebas.cliff([2, 3, 4, 10], [1, 2, 3, 4]) == (pytest.approx(0.4375), "mediana")


def test_bca_cubre_la_media_y_respeta_conglomerados():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(200), 3)                    # 200 conglomerados de 3 filas
    x = np.repeat(rng.normal(size=200), 3) + rng.normal(scale=0.1, size=600)
    est, inf, sup = pruebas.bca([(g, x)], lambda m: float(np.mean(m[0])), B=400)
    assert inf < est < sup
    # el ancho refleja 200 unidades independientes, no 600: cerca de 2 · 1,96 · 1/√200
    assert 0.2 < sup - inf < 0.35
