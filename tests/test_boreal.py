"""BOREAL, el modelo original (original/modelo_original_otif.py sobre original/arboles_propios.py): aprende un caso
sintético con estructura conocida, es reproducible, rechaza entradas inválidas, se clona como un estimador de
scikit-learn, respeta la monotonía en el corte y no usa bibliotecas de aprendizaje."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

ORIGINAL = Path(__file__).resolve().parents[1] / "original"
sys.path.insert(0, str(ORIGINAL))

from arboles_propios import BoostingHistograma, Discretizador  # noqa: E402
from modelo_original_otif import Boreal  # noqa: E402

ENT = ["proveedor", "material", "sp_creador", "creador_oc"]


@pytest.fixture(scope="module")
def sintetico():
    """Retraso log-normal con dos numéricas, una categórica y un efecto real por proveedor."""
    rng = np.random.default_rng(7)
    n = 6000
    prov = rng.integers(0, 120, n)
    efecto = rng.normal(0, 1.0, 120)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(list("abcd"), n)
    dias = np.exp(0.9 * x1 - 0.6 * x2 + 0.5 * (cat == "a") + efecto[prov] + rng.normal(0, 0.8, n)) - 1.5
    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": pd.Categorical(cat), "proveedor": prov.astype(str),
                      "material": "m", "sp_creador": "s", "creador_oc": "c"})
    y = (dias > 1).astype(int)
    tr = np.arange(n) < 4500
    return X, y, dias, tr, efecto


def _rapido(**kw):
    return Boreal(n_estimators=60, arboles_efectos=25, vueltas=2, num_leaves=15, min_child_samples=20,
                  learning_rate=0.1, **kw)


@pytest.fixture(scope="module")
def ajustado(sintetico):
    X, y, dias, tr, _ = sintetico
    return _rapido().fit(X[tr], y[tr], dias=dias[tr])


def test_aprende_el_caso_sintetico(sintetico, ajustado):
    X, y, _, tr, efecto = sintetico
    auc = roc_auc_score(y[~tr], ajustado.predict_proba(X[~tr])[:, 1])
    assert auc > 0.85
    niveles = ajustado.motor_.niveles["proveedor"].valores.astype(int)
    assert np.corrcoef(ajustado.motor_.b["proveedor"], efecto[niveles])[0, 1] > 0.8   # recupera los efectos reales


def test_reproducible_con_la_misma_semilla(sintetico, ajustado):
    X, y, dias, tr, _ = sintetico
    otro = _rapido().fit(X[tr], y[tr], dias=dias[tr])
    assert np.array_equal(ajustado.decision_function(X[~tr]), otro.decision_function(X[~tr]))


def test_monotono_en_el_corte(sintetico, ajustado):
    X, *_ = sintetico
    sin = X.drop(columns=ENT)
    assert ajustado.motor_.fraccion_no_monotona(sin, {e: X[e].to_numpy() for e in ENT}) == 0.0


def test_clone_y_parametros(ajustado):
    c = clone(ajustado)
    assert c.get_params() == ajustado.get_params()
    assert not hasattr(c, "motor_")


def test_nivel_nuevo_recibe_el_efecto_medio(sintetico, ajustado):
    X, *_ = sintetico
    a = X.iloc[:5].copy()
    b = a.copy()
    b["proveedor"] = "nunca visto"
    d = ajustado.motor_.desplazamiento({e: b[e].to_numpy() for e in ENT})
    assert np.allclose(d, 0.0)


def test_entradas_invalidas(sintetico):
    X, y, dias, tr, _ = sintetico
    with pytest.raises(ValueError):
        _rapido().fit(X[tr], np.where(y[tr] == 1, 2, 0) + (np.arange(tr.sum()) % 3 == 0), dias=dias[tr])
    with pytest.raises(ValueError):
        _rapido().fit(X[tr].drop(columns="proveedor"), y[tr])
    with pytest.raises(TypeError):
        _rapido().fit(X[tr].drop(columns=["cat"] + ENT).to_numpy(), y[tr])
    with pytest.raises(ValueError):
        _rapido().fit(X[tr], y[tr][:-1])
    with pytest.raises(AttributeError):
        _rapido().predict_proba(X)


def test_sin_dias_es_un_clasificador_de_un_corte(sintetico):
    X, y, _, tr, _ = sintetico
    m = _rapido().fit(X[tr], y[tr])
    assert m.motor_.cortes == [0]
    assert roc_auc_score(y[~tr], m.predict_proba(X[~tr])[:, 1]) > 0.8
    assert set(np.unique(m.predict(X[~tr]))) <= {0, 1}


def test_regresion_cuadratica_baja_el_error():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 + rng.normal(0, 0.1, 3000)
    cols = [x[:, j] for j in range(3)]
    T = Discretizador().fit(cols, [False] * 3).transform(cols)
    m = BoostingHistograma(n_estimators=100, learning_rate=0.1, num_leaves=15, objetivo="cuadratica").fit(T, y, [False] * 3)
    assert np.mean((m.predict_raw(T) - y) ** 2) < 0.1 * np.var(y)


def test_solo_numpy_y_scipy():
    """El algoritmo no importa bibliotecas de aprendizaje (de scikit-learn, solo las clases base)."""
    permitidos = {"numpy", "scipy", "heapq", "os", "multiprocessing", "arboles_propios", "sklearn.base"}
    for archivo in ("arboles_propios.py", "modelo_original_otif.py"):
        arbol = ast.parse((ORIGINAL / archivo).read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres = [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                nombres = [nodo.module]
            else:
                continue
            for nombre in nombres:
                assert nombre in permitidos or nombre.split(".")[0] in {"numpy", "scipy"}, f"{archivo} importa {nombre}"


def _esperado_condicional(arbol, t, S):
    """E[f(x) | x_S] con la dependencia del camino (Lundberg 2020, algoritmo 1), por recursión."""
    def rec(nodo):
        if arbol.izq[nodo] < 0:
            return arbol.valor[nodo]
        if arbol.var[nodo] in S:
            izq = bool(arbol.izquierda(np.array([nodo]), np.array([t[arbol.var[nodo]]]))[0])
            return rec(arbol.izq[nodo] if izq else arbol.der[nodo])
        a, b = arbol.izq[nodo], arbol.der[nodo]
        return (arbol.cobertura[a] * rec(a) + arbol.cobertura[b] * rec(b)) / arbol.cobertura[nodo]
    return rec(0)


def test_treeshap_igual_a_shapley_por_fuerza_bruta():
    from itertools import combinations
    from math import factorial
    rng = np.random.default_rng(11)
    n, p = 800, 4
    x = rng.normal(size=(n, p))
    x[rng.random((n, p)) < 0.1] = np.nan
    cat = rng.choice(list("abcdefg"), n)
    cols = [x[:, j] for j in range(p)] + [cat]
    es_cat = [False] * p + [True]
    y = (np.nan_to_num(x[:, 0]) + np.nan_to_num(x[:, 1] * x[:, 2]) + (cat < "d") + rng.normal(0, 0.5, n) > 0.5).astype(int)
    T = Discretizador().fit(cols, es_cat).transform(cols)
    m = BoostingHistograma(n_estimators=3, num_leaves=8, min_child_samples=10, subsample=0.8, cat_smooth=1.0,
                           min_data_per_group=5, learning_rate=0.3).fit(T, y, es_cat)
    fi, esperado = m.shap(T[:, :20])
    assert np.allclose(fi.sum(1) + esperado, m.predict_raw(T[:, :20]))       # precisión local
    M = T.shape[0]
    for r in range(5):
        t = T[:, r]
        exacto = np.zeros(M)
        for a in m.arboles_:
            for i in range(M):
                otros = [j for j in range(M) if j != i]
                for k in range(M):
                    for S in combinations(otros, k):
                        w = factorial(k) * factorial(M - k - 1) / factorial(M)
                        exacto[i] += w * (_esperado_condicional(a, t, set(S) | {i}) - _esperado_condicional(a, t, set(S)))
        assert np.allclose(fi[r], exacto, atol=1e-10)


def test_funciona_en_pipeline_y_cross_val_score(sintetico):
    # el Entregable 3 pide que el modelo propio se integre con Pipeline y cross_val_score
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline

    from src.original_model import Boreal as BorealSrc
    X, y, _, _, _ = sintetico
    assert BorealSrc is Boreal
    pipe = Pipeline([("modelo", _rapido())])
    s = cross_val_score(pipe, X, y, cv=KFold(3, shuffle=True, random_state=0), scoring="roc_auc")
    assert len(s) == 3 and (s > 0.8).all()
