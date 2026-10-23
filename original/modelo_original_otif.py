"""Modelo original para OTIF: boosting de arboles con efectos aleatorios cruzados por entidad sobre la reduccion ordinal
de los dias de retraso. Lo usa evaluar_original_otif.py.

Formulacion (fichas en revision_literatura/fichas_otif/):
    P(dias_retraso > c_k | x) = sigmoide( f(x, c_k) + b_proveedor + b_material + b_sp_creador + b_creador_oc )
  - Reduccion ordinal (Li y Lin 2006): cada posicion se repite una vez por corte c_k en {0, 1, 5, 15} con la etiqueta
    [dias > c_k] y el corte como una variable mas; pesos iguales, que con costo absoluto son los del articulo. Se predice
    en c = 1, que es y_tarde.
  - Umbrales ordenados: restriccion monotona decreciente de f en el corte, para que P(d > 0) >= P(d > 1) >= ... en cada
    posicion (la ficha de Li y Lin, camino b). Asi el arbol conserva las interacciones x por corte.
  - Efectos aleatorios cruzados (Griesbach 2021, Knieper 2025): intercepto b_e por nivel de cada entidad, con previa
    N(0, s2_e), igual en todos los cortes, que conserva la monotonia. Se estiman por bloques alternados: un boosting con
    los efectos como desplazamiento del logit, y luego pasos de Newton por entidad con s2_e por EM con la curvatura de
    Laplace. Al estimar se agrega un efecto por OC, que absorbe la dependencia entre posiciones de la misma orden y se
    descarta al predecir. Los efectos se estiman con el corte de interes (c = 1): la suma de perdidas de las copias no es
    una verosimilitud, porque las etiquetas de una posicion dependen de un solo d (ficha de Li y Lin, 6.3).
  - Correccion C (Griesbach 2021, apendice) en su forma con columna de unos: cada efecto se centra en cada barrido, para
    que no arrastre un nivel medio que le toca al arbol. Por eso un nivel nuevo, que recibe 0, recibe el efecto medio.
  - Las te_* de las cuatro entidades salen de las variables: su lugar lo toma el efecto aleatorio.
Lo que el articulo no trae y es propio: respuesta binaria con efectos cruzados de cuatro entidades y arboles, y la
combinacion con la reduccion ordinal.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

ENTIDADES_RE = ["proveedor", "material", "sp_creador", "creador_oc"]
CORTES = [0, 1, 5, 15]
CORTE_OBJETIVO = 1
SEMILLA = 2026
# hiperparametros del LightGBM del modelo actual (Optuna, ensayo 33), solo para estimar los efectos
PRM_EFECTOS = dict(learning_rate=0.01698, num_leaves=110, min_child_samples=143, subsample=0.981, subsample_freq=1,
                   colsample_bytree=0.3655, reg_lambda=0.007801, reg_alpha=2.399, cat_smooth=1.542, min_data_per_group=28)


def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class BoostingMixtoOrdinal:
    def __init__(self, prm, arboles=3000, monotono=True, centrar=True, vueltas=3):
        self.prm, self.arboles, self.monotono, self.centrar, self.vueltas = dict(prm, subsample_freq=1), arboles, monotono, centrar, vueltas

    # ---- efectos aleatorios ------------------------------------------------------------------------------------------
    def _estimar_efectos(self, X, y, grupos, oc, base):
        cods = {e: pd.Index(grupos[e].astype(str).unique()) for e in ENTIDADES_RE}
        idx = {e: cods[e].get_indexer(grupos[e].astype(str)) for e in ENTIDADES_RE}
        idx["oc_id"] = pd.factorize(oc)[0]
        k = {e: int(i.max()) + 1 for e, i in idx.items()}
        b = {e: np.zeros(k[e]) for e in idx}
        s2 = {e: 0.5 for e in idx}
        desp = lambda excepto=None: base + sum(b[e][idx[e]] for e in idx if e != excepto)
        for _ in range(self.vueltas):
            f = lgb.LGBMClassifier(n_estimators=474, random_state=SEMILLA, verbose=-1, n_jobs=-1, **PRM_EFECTOS).fit(
                X, y, init_score=desp()).predict(X, raw_score=True)
            for _ in range(2):
                for e in idx:
                    otro = f + desp(e)
                    for _ in range(3):
                        p = sigmoide(otro + b[e][idx[e]])
                        h = np.bincount(idx[e], p * (1 - p), k[e]) + 1 / s2[e]
                        b[e] = b[e] + (np.bincount(idx[e], y - p, k[e]) - b[e] / s2[e]) / h
                    if self.centrar:
                        b[e] = b[e] - b[e].mean()          # correccion C con la columna de unos
                    s2[e] = float(np.mean(b[e] ** 2 + 1 / h))
        self.niveles = cods
        self.b = {e: b[e] for e in ENTIDADES_RE}           # el de OC solo sirvio para estimar sin sobreajuste
        self.s2 = s2
        return sum(b[e][idx[e]] for e in ENTIDADES_RE)

    def desplazamiento(self, grupos):
        """Suma de los efectos de cada fila; un nivel que no estaba al entrenar recibe 0."""
        d = np.zeros(len(next(iter(grupos.values()))))
        for e in ENTIDADES_RE:
            i = self.niveles[e].get_indexer(grupos[e].astype(str))
            d += np.where(i >= 0, self.b[e][np.maximum(i, 0)], 0.0)
        return d

    # ---- ajuste --------------------------------------------------------------------------------------------------------
    @staticmethod
    def _ampliar(X, dias):
        Xa = pd.concat([X.assign(corte=float(c)) for c in CORTES], ignore_index=True)
        return Xa, np.concatenate([(dias > c).astype(int) for c in CORTES])

    def fit(self, X, dias, grupos, oc, validacion=None, base=None, efectos_ent=None):
        """X sin las te_* de ENTIDADES_RE; grupos: dict entidad -> serie; validacion = (X, y, grupos[, base]) para parar
        temprano. base: logit de otro modelo como desplazamiento fijo (el modelo aprende lo que ese no explica; debe venir
        fuera de pliegue en el entrenamiento). Sin base es el modelo original solo.
        efectos_ent: si se da, es la suma de efectos que ven los ARBOLES en cada fila de entrenamiento en lugar de los
        efectos estimados con todo (sondear_frio_otif.py: efectos apagados en filas frias u ordenados en el tiempo); los
        efectos con todo el entrenamiento se estiman igual y son los que se usan al predecir."""
        dias = np.asarray(dias)
        base = np.zeros(len(dias)) if base is None else np.asarray(base, float)
        d = base + self._estimar_efectos(X, (dias > CORTE_OBJETIVO).astype(int), grupos, oc, base)
        if efectos_ent is not None:
            d = base + np.asarray(efectos_ent, float)
        Xa, ya = self._ampliar(X, dias)
        prm = dict(self.prm)
        if self.monotono:
            prm["monotone_constraints"] = [0] * X.shape[1] + [-1]
        self.modelo = lgb.LGBMClassifier(n_estimators=self.arboles, random_state=SEMILLA, verbose=-1, n_jobs=-1, **prm)
        extra = {}
        if validacion is not None:
            Xv, yv, gv = validacion[:3]
            bv = validacion[3] if len(validacion) > 3 else 0.0
            extra = dict(eval_set=[(Xv.assign(corte=float(CORTE_OBJETIVO)), yv)], eval_init_score=[self.desplazamiento(gv) + bv],
                         eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        self.modelo.fit(Xa, ya, init_score=np.tile(d, len(CORTES)), **extra)
        self.arboles_usados = int(self.modelo.best_iteration_ or self.arboles)
        return self

    def logit(self, X, grupos, corte=CORTE_OBJETIVO, base=0.0):
        return (self.modelo.predict(X.assign(corte=float(corte)), raw_score=True, num_iteration=self.arboles_usados)
                + self.desplazamiento(grupos) + base)

    def predict_proba(self, X, grupos, base=0.0):
        return sigmoide(self.logit(X, grupos, base=base))

    def fraccion_no_monotona(self, X, grupos):
        """Fraccion de filas con P(d > c_k) < P(d > c_k+1) en algun par de cortes; con la restriccion debe dar 0."""
        z = np.c_[[self.logit(X, grupos, c) for c in CORTES]].T
        return float(np.mean((np.diff(z, axis=1) > 1e-9).any(axis=1)))
