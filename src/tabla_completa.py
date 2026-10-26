"""La tabla maestra que pide el Entregable 3: las 112 combinaciones (``resultados/tabla_maestra.csv``) más una fila del
modelo original.

Las 112 se miden con la media de los ocho bloques externos de la validación anidada. BOREAL no corrió esos ocho bloques
(cada celda tarda horas): su fila es la medida de la validación 2025 con entrenamiento 2021-2024, la misma que fijó su
Platt y su umbral (``original/config_2x2/prop_nuevas.json``), y la columna ``medida`` lo dice. Su comparación
con el XGBoost elegido está en la prueba 2026 (capítulo del modelo original), no en esta tabla.

Uso: ``python -m src.tabla_completa`` (deja ``resultados/tabla_maestra_completa.csv``).
"""
import json

import pandas as pd

from . import datos


def main():
    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv")
    t.insert(3, "medida", "media de 8 bloques externos (2022-S1 a 2025-S2)")
    orig = datos.RAIZ / "original"
    cfg = json.loads((orig / "config_2x2" / "prop_nuevas.json").read_text(encoding="utf-8"))
    hp = json.loads((orig / "original_congelado.json").read_text(encoding="utf-8"))
    fila = {"modelo": "BOREAL (modelo original)", "balanceo": "ninguno",
            "optimizador": "Optuna en la validación, congelado (capítulo del modelo original)",
            "medida": "validación 2025, entrenamiento 2021-2024", "bloques": 1, "AUC": cfg["auc_validacion"],
            "F1": cfg["f1_validacion"], "hiperparametros_finales": json.dumps(dict(hp["prm"], cortes=[0, 1, 5, 15],
                                                                                   monotono=hp["monotono"],
                                                                                   centrar=hp["centrar"])),
            "semilla": datos.SEMILLA}
    t = pd.concat([t, pd.DataFrame([fila])], ignore_index=True)
    t.to_csv(datos.RAIZ / "resultados" / "tabla_maestra_completa.csv", index=False)
    print(t.tail(2).T.to_string())


if __name__ == "__main__":
    main()
