"""XGBoost con el escalado de Platt congelado en la validación 2025: lo que sirve el tablero y lo que se evaluó en 2026.

La probabilidad es σ(a · margen + b), con ``margen`` el logit crudo de XGBoost y (a, b) los de ``config_2x2/xgb136.json``
(ajustados en 2025, antes de mirar la prueba). ``get_booster`` deja a la vista el XGBoost de adentro, para que el tablero
saque las contribuciones exactas de árbol (``pred_contribs``), que están en la escala del margen, antes de Platt.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def denso(X):
    """Matriz densa float32. XGBoost toma los ceros de una matriz dispersa como vacíos, y el one-hot sale disperso: sin
    esto el modelo no sería el de los experimentos, que entrenan con matrices densas."""
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


class XGBoostPlatt(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, platt=(1.0, 0.0), semilla=2026):
        self.params = params
        self.platt = platt
        self.semilla = semilla

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.modelo_ = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=self.semilla, eval_metric="logloss",
                                     **(self.params or {}))
        self.modelo_.fit(denso(X), y)
        self.classes_ = self.modelo_.classes_
        return self

    def margen(self, X):
        return self.modelo_.predict(denso(X), output_margin=True)

    def predict_proba(self, X):
        a, b = self.platt
        p = 1 / (1 + np.exp(-(a * self.margen(X) + b)))
        return np.column_stack([1 - p, p])

    def get_booster(self):
        return self.modelo_.get_booster()
