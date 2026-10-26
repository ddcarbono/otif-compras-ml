"""BOREAL (Boosting Ordinal de Retraso con Efectos ALeatorios): el modelo original para OTIF. Boosting de árboles con
efectos aleatorios cruzados por entidad sobre la reducción ordinal de los días de retraso. Implementación propia: el
boosting es `arboles_propios.BoostingHistograma` (solo numpy); de scikit-learn solo se heredan `BaseEstimator` y
`ClassifierMixin` para que la clase se comporte como un estimador (clone, get_params, score).

Formulación (fichas en revision_literatura/fichas_otif/):
    P(dias_retraso > c_k | x) = sigmoide( f(x, c_k) + b_proveedor + b_material + b_sp_creador + b_creador_oc )
  - Reducción ordinal (Li y Lin 2006): cada posición se repite una vez por corte c_k en {0, 1, 5, 15} con la etiqueta
    [dias > c_k] y el corte como una variable más; pesos iguales. Se predice en c = 1, que es y_tarde.
  - Umbrales ordenados: restricción monótona decreciente de f en el corte, para que P(d > 0) >= P(d > 1) >= ... en cada
    posición (Li y Lin, camino b). Así el árbol conserva las interacciones x por corte.
  - Efectos aleatorios cruzados (Griesbach 2021, Knieper 2025): intercepto b_e por nivel de cada entidad, con previa
    N(0, s2_e), igual en todos los cortes, que conserva la monotonía. Se estiman por bloques alternados: un boosting con
    los efectos como desplazamiento del logit, y luego pasos de Newton por entidad con s2_e por EM con la curvatura de
    Laplace. Al estimar se agrega un efecto por OC, que absorbe la dependencia entre posiciones de la misma orden y se
    descarta al predecir. Los efectos se estiman con el corte de interés (c = 1).
  - Corrección C (Griesbach 2021, apéndice) con columna de unos: cada efecto se centra en cada barrido, para que no
    arrastre un nivel medio que le toca al árbol. Un nivel nuevo recibe 0, es decir, el efecto medio.
  - Las te_* de las cuatro entidades salen de las variables: su lugar lo toma el efecto aleatorio.

Pseudocódigo de `BoostingMixtoOrdinal.fit`:
    b_e <- 0 para cada entidad e (y para la OC);  s2_e <- 0,5
    repetir `vueltas` veces:
        f <- boosting(X, y_tarde, desplazamiento = sum_e b_e)                 # PRM_EFECTOS, 474 árboles
        repetir 2 veces, para cada entidad e:
            3 pasos de Newton:  b_e <- b_e + (sum (y - p) - b_e / s2_e) / (sum p(1 - p) + 1 / s2_e)
            centrar b_e;  s2_e <- media(b_e^2 + 1 / curvatura)
    ampliar: cada fila se repite con corte c en CORTES y etiqueta [dias > c]
    f <- boosting(X ampliado, desplazamiento = sum_e b_e, monotonía decreciente en el corte)
Complejidad: la del boosting (ver arboles_propios) por (vueltas + 1) ajustes, el último sobre 4n filas; los efectos
cuestan O(n) por paso de Newton.
"""
import multiprocessing as mp
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from arboles_propios import BoostingHistograma, Discretizador, sigmoide

ENTIDADES_RE = ["proveedor", "material", "sp_creador", "creador_oc"]
CORTES = [0, 1, 5, 15]
CORTE_OBJETIVO = 1
SEMILLA = 2026
# hiperparámetros del boosting auxiliar con que se estiman los efectos (Optuna, ensayo 33 del modelo de referencia)
PRM_EFECTOS = dict(learning_rate=0.01698, num_leaves=110, min_child_samples=143, subsample=0.981,
                   colsample_bytree=0.3655, reg_lambda=0.007801, reg_alpha=2.399, cat_smooth=1.542, min_data_per_group=28)
# BOREAL_ESCALA_ARBOLES (por defecto 1) achica todos los conteos de árboles para una prueba de humo del guion
ESCALA = float(os.environ.get("BOREAL_ESCALA_ARBOLES", "1"))
ARBOLES_EFECTOS = 474


def escalar(n):
    return max(2, int(round(n * ESCALA)))


def es_categorica(x):
    return getattr(x, "dtype", None) is not None and (str(x.dtype) in ("category", "object") or x.dtype == object)


def columnas(X):
    """Lista de arreglos 1-D y si cada uno es categórico, de un DataFrame (o de una matriz numérica)."""
    if hasattr(X, "columns"):
        cols = [X[c] for c in X.columns]
        return [np.asarray(c.astype(object) if es_categorica(c) else c) for c in cols], [es_categorica(c) for c in cols]
    X = np.asarray(X, dtype=float)
    return [X[:, j] for j in range(X.shape[1])], [False] * X.shape[1]


class Niveles:
    """Código entero de cada nivel de una entidad, con los niveles vistos al entrenar; uno nuevo recibe -1."""

    def fit(self, x):
        self.valores = np.unique(np.asarray(x).astype(str))
        return self

    def transform(self, x):
        x = np.asarray(x).astype(str)
        i = np.searchsorted(self.valores, x)
        i = np.minimum(i, len(self.valores) - 1)
        return np.where(self.valores[i] == x, i, -1)


class BoostingMixtoOrdinal:
    """El motor de BOREAL con la interfaz que usan los guiones de evaluación (X sin las entidades; grupos aparte)."""

    def __init__(self, prm, arboles=3000, monotono=True, centrar=True, vueltas=3, cortes=None, corte_objetivo=None,
                 prm_efectos=None, arboles_efectos=ARBOLES_EFECTOS, semilla=SEMILLA):
        self.prm, self.arboles, self.monotono, self.centrar, self.vueltas = dict(prm), arboles, monotono, centrar, vueltas
        self.cortes = list(CORTES if cortes is None else cortes)
        self.corte_objetivo = CORTE_OBJETIVO if corte_objetivo is None else corte_objetivo
        self.prm_efectos = dict(PRM_EFECTOS if prm_efectos is None else prm_efectos)
        self.arboles_efectos, self.semilla = arboles_efectos, semilla

    # ---- datos ---------------------------------------------------------------------------------------------------------
    def _discretizar(self, X):
        cols, cat = columnas(X)
        if not hasattr(self, "disc"):
            self.disc, self.cat = Discretizador().fit(cols, cat), cat
        return self.disc.transform(cols)

    # ---- efectos aleatorios --------------------------------------------------------------------------------------------
    def _estimar_efectos(self, X, y, grupos, oc, base, T=None):
        T = self._discretizar(X) if T is None else T
        y = np.asarray(y, dtype=float)
        self.niveles = {e: Niveles().fit(grupos[e]) for e in grupos}
        idx = {e: self.niveles[e].transform(grupos[e]) for e in grupos}
        idx["oc_id"] = np.unique(np.asarray(oc).astype(str), return_inverse=True)[1]
        k = {e: int(i.max()) + 1 for e, i in idx.items()}
        b = {e: np.zeros(k[e]) for e in idx}
        s2 = {e: 0.5 for e in idx}
        desp = lambda excepto=None: base + sum(b[e][idx[e]] for e in idx if e != excepto)
        for _ in range(self.vueltas):
            aux = BoostingHistograma(n_estimators=escalar(self.arboles_efectos), semilla=self.semilla, **self.prm_efectos)
            f = aux.fit(T, y, self.cat, init=desp()).predict_raw(T)
            for _ in range(2):
                for e in idx:
                    otro = f + desp(e)
                    for _ in range(3):
                        p = sigmoide(otro + b[e][idx[e]])
                        h = np.bincount(idx[e], p * (1 - p), k[e]) + 1 / s2[e]
                        b[e] = b[e] + (np.bincount(idx[e], y - p, k[e]) - b[e] / s2[e]) / h
                    if self.centrar:
                        b[e] = b[e] - b[e].mean()          # corrección C con la columna de unos
                    s2[e] = float(np.mean(b[e] ** 2 + 1 / h))
        self.b = {e: b[e] for e in grupos}                 # el de OC solo sirvió para estimar sin sobreajuste
        self.s2 = s2
        return sum(b[e][idx[e]] for e in grupos)

    def desplazamiento(self, grupos):
        """Suma de los efectos de cada fila; un nivel que no estaba al entrenar recibe 0."""
        d = np.zeros(len(next(iter(grupos.values()))))
        for e in self.b:
            i = self.niveles[e].transform(grupos[e])
            d += np.where(i >= 0, self.b[e][np.maximum(i, 0)], 0.0)
        return d

    # ---- ajuste --------------------------------------------------------------------------------------------------------
    def _ampliar(self, T, corte):
        """T con la fila del corte agregada (el corte es una numérica más, tramo = posición del corte + 1)."""
        return np.vstack([T, np.full((1, T.shape[1]), self.cortes.index(corte) + 1, dtype=np.uint8)])

    def fit(self, X, dias, grupos, oc, base=None, efectos_ent=None):
        """X sin las te_* de las entidades; grupos: dict entidad -> serie; base: logit fijo de otro modelo;
        efectos_ent: suma de efectos que ven los ÁRBOLES en cada fila (efectos ordenados en el tiempo); los efectos
        con todo el entrenamiento se estiman igual y son los que se usan al predecir."""
        dias = np.asarray(dias, dtype=float)
        T = self._discretizar(X)
        base = np.zeros(len(dias)) if base is None else np.asarray(base, float)
        d = base + self._estimar_efectos(X, (dias > self.corte_objetivo).astype(int), grupos, oc, base, T=T)
        if efectos_ent is not None:
            d = base + np.asarray(efectos_ent, float)
        Ta = np.hstack([self._ampliar(T, c) for c in self.cortes])
        ya = np.concatenate([(dias > c).astype(int) for c in self.cortes])
        mono = [0] * T.shape[0] + ([-1] if self.monotono and len(self.cortes) > 1 else [0])
        self.modelo = BoostingHistograma(n_estimators=escalar(self.arboles), semilla=self.semilla, monotonia=mono, **self.prm)
        self.modelo.fit(Ta, ya, self.cat + [False], init=np.tile(d, len(self.cortes)))
        self.arboles_usados = escalar(self.arboles)
        return self

    def logit(self, X, grupos, corte=None, base=0.0):
        corte = self.corte_objetivo if corte is None else corte
        T = self._ampliar(self._discretizar(X), corte)
        return self.modelo.predict_raw(T) + self.desplazamiento(grupos) + base

    def predict_proba(self, X, grupos, base=0.0):
        return sigmoide(self.logit(X, grupos, base=base))

    def fraccion_no_monotona(self, X, grupos):
        """Fracción de filas con P(d > c_k) < P(d > c_k+1) en algún par de cortes; con la restricción debe dar 0."""
        z = np.c_[[self.logit(X, grupos, c) for c in self.cortes]].T
        return float(np.mean((np.diff(z, axis=1) > 1e-9).any(axis=1)))


_CTX = {}


def _efectos_de_un_periodo(t):
    c = _CTX
    antes, este = c["periodo"] < t, c["periodo"] == t
    aux = BoostingMixtoOrdinal(c["prm"], centrar=c["centrar"], prm_efectos=c["prm_efectos"])
    aux._estimar_efectos(c["X"].iloc[np.flatnonzero(antes)] if hasattr(c["X"], "iloc") else c["X"][antes], c["y"][antes],
                         {e: np.asarray(v)[antes] for e, v in c["grupos"].items()}, np.asarray(c["oc"])[antes],
                         np.zeros(antes.sum()))
    return t, aux.desplazamiento({e: np.asarray(v)[este] for e, v in c["grupos"].items()})


def efectos_ordenados(X, y, grupos, oc, periodo, prm, centrar=True, prm_efectos=None, procesos=None, aviso=print):
    """Efectos que ven los árboles en cada fila, estimados solo con los periodos anteriores (el primero recibe 0). Cada
    periodo es independiente: con fork (Linux) se reparten entre procesos."""
    periodo = np.asarray(periodo)
    _CTX.update(X=X, y=np.asarray(y), grupos=grupos, oc=oc, periodo=periodo, prm=prm, centrar=centrar,
                prm_efectos=prm_efectos)
    ts = sorted(set(periodo.tolist()))[1:]
    d = np.zeros(len(periodo))
    procesos = procesos or int(os.environ.get("BOREAL_PROCESOS", "1"))
    if procesos > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(min(procesos, len(ts))) as pool:
            resultados = pool.map(_efectos_de_un_periodo, ts[::-1])     # los más largos primero
    else:
        resultados = map(_efectos_de_un_periodo, ts)
    for t, v in resultados:
        d[periodo == t] = v
        aviso(f"   efectos ordenados para {t} con {(periodo < t).sum()} filas anteriores")
    _CTX.clear()
    return d


class Boreal(BaseEstimator, ClassifierMixin):
    """BOREAL con la interfaz de scikit-learn.

    fit(X, y, dias=None): X es un DataFrame con las predictoras y las columnas de `entidades` (efectos aleatorios) y,
    si se da `columna_oc`, la de la orden (solo para estimar). y es la etiqueta binaria (0/1); `dias` es la medida
    continua de la que sale (y = dias > corte_objetivo). Sin `dias`, la reducción ordinal usa un solo corte.
    predict_proba(X) devuelve P(y = 1) en el corte objetivo, sin calibrar."""

    def __init__(self, entidades=("proveedor", "material", "sp_creador", "creador_oc"), columna_oc=None,
                 cortes=(0, 1, 5, 15), corte_objetivo=1, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0, cat_smooth=10.0,
                 min_data_per_group=100, n_estimators=200, arboles_efectos=100, vueltas=2, monotono=True, centrar=True,
                 semilla=2026):
        self.entidades, self.columna_oc, self.cortes, self.corte_objetivo = entidades, columna_oc, cortes, corte_objetivo
        self.learning_rate, self.num_leaves, self.min_child_samples = learning_rate, num_leaves, min_child_samples
        self.subsample, self.colsample_bytree, self.reg_lambda, self.reg_alpha = subsample, colsample_bytree, reg_lambda, reg_alpha
        self.cat_smooth, self.min_data_per_group, self.n_estimators = cat_smooth, min_data_per_group, n_estimators
        self.arboles_efectos, self.vueltas, self.monotono, self.centrar, self.semilla = arboles_efectos, vueltas, monotono, centrar, semilla

    def _partir(self, X):
        if not hasattr(X, "columns"):
            raise TypeError("Boreal espera un DataFrame con las columnas de las entidades")
        faltan = [c for c in self.entidades if c not in X.columns]
        if faltan:
            raise ValueError(f"faltan las columnas de entidades {faltan}")
        quitar = list(self.entidades) + ([self.columna_oc] if self.columna_oc else [])
        return X[[c for c in X.columns if c not in quitar]], {e: X[e].to_numpy() for e in self.entidades}

    def fit(self, X, y, dias=None):
        Xp, grupos = self._partir(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(Xp):
            raise ValueError("y debe ser un vector con una etiqueta por fila")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2 or not set(self.classes_.tolist()) <= {0, 1}:
            raise ValueError("BOREAL es un clasificador binario: y debe tener las clases 0 y 1")
        if dias is None:
            dias, cortes = y.astype(float), [0]
            corte = 0
        else:
            dias = np.asarray(dias, dtype=float)
            if np.any(np.isnan(dias)):
                raise ValueError("dias no puede tener vacíos")
            cortes, corte = list(self.cortes), self.corte_objetivo
        oc = X[self.columna_oc].to_numpy() if self.columna_oc else np.arange(len(Xp))
        prm = dict(learning_rate=self.learning_rate, num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
                   subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                   reg_alpha=self.reg_alpha, cat_smooth=self.cat_smooth, min_data_per_group=self.min_data_per_group)
        self.motor_ = BoostingMixtoOrdinal(prm, arboles=self.n_estimators, monotono=self.monotono, centrar=self.centrar,
                                           vueltas=self.vueltas, cortes=cortes, corte_objetivo=corte, prm_efectos=prm,
                                           arboles_efectos=self.arboles_efectos, semilla=self.semilla)
        self.motor_.fit(Xp, dias, grupos, oc)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        if not hasattr(self, "motor_"):
            raise AttributeError("BOREAL no está ajustado: llame a fit primero")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X trae {X.shape[1]} columnas y el modelo se ajustó con {self.n_features_in_}")
        Xp, grupos = self._partir(X)
        return self.motor_.logit(Xp, grupos)

    def predict_proba(self, X):
        p = sigmoide(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
