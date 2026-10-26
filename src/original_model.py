"""El modelo original, BOREAL, con la ruta que pide el Entregable 3 (``src/original_model.py``).

La implementación vive en ``original/`` porque con esos nombres de módulo se guardó el modelo que sirve el tablero
(``artifacts/original_servido.joblib``); este archivo solo la expone:

- ``Boreal``: la clase de scikit-learn (``BaseEstimator`` + ``ClassifierMixin``), en ``original/modelo_original_otif.py``;
- ``BoostingMixtoOrdinal`` y ``efectos_ordenados``: la estimación por bloques de los efectos y su versión ordenada en
  el tiempo, en el mismo módulo;
- ``BoostingHistograma`` y ``Discretizador``: el aprendiz base de árboles de histograma con TreeSHAP, en
  ``original/arboles_propios.py``.

Todo está escrito con numpy y scipy. Uso::

    from src.original_model import Boreal
    m = Boreal(entidades=["proveedor", "material", "sp_creador", "creador_oc"], columna_oc="oc_id")
    m.fit(X, y, dias=dias_de_retraso)
    m.predict_proba(X_nuevo)[:, 1]
"""
import sys
from pathlib import Path

_ORIGINAL = Path(__file__).resolve().parents[1] / "original"
if str(_ORIGINAL) not in sys.path:
    sys.path.insert(0, str(_ORIGINAL))

from arboles_propios import BoostingHistograma, Discretizador  # noqa: E402
from modelo_original_otif import Boreal, BoostingMixtoOrdinal, efectos_ordenados  # noqa: E402

__all__ = ["Boreal", "BoostingMixtoOrdinal", "efectos_ordenados", "BoostingHistograma", "Discretizador"]
