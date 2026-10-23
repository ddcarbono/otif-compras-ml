"""Junta las dos rondas de conjuntos publicos (publicos_otif.csv y publicos_otif_ronda2.csv): p de DeLong ajustado por Holm
sobre los siete conjuntos y Wilcoxon de rangos con signo sobre las siete diferencias de AUC (K3 - XGBoost).

Uso: python resumir_publicos_otif.py   (deja publicos_otif_resumen.txt y .csv aqui)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

AQUI = Path(__file__).parent
d = pd.concat([pd.read_csv(AQUI / "publicos_otif.csv").assign(ronda=1), pd.read_csv(AQUI / "publicos_otif_ronda2.csv").assign(ronda=2)])
p = d.p_delong.to_numpy()
orden, m, acum, holm = np.argsort(p), len(p), 0.0, np.empty(len(p))
for k, i in enumerate(orden):
    acum = max(acum, (m - k) * p[i])
    holm[i] = min(1.0, acum)
d["p_holm"] = holm.round(4)
d["veredicto"] = np.where(d.p_holm < 0.05, np.where(d.dif > 0, "K3 gana", "K3 pierde"), "empate")
w = wilcoxon(d.dif)
texto = (d[["ronda", "conjunto", "n", "tasa", "auc_xgb", "auc_k3", "dif", "ic_lo", "ic_hi", "p_delong", "p_holm", "veredicto",
            "f1_xgb", "f1_k3"]].to_string(index=False)
         + f"\n\nK3 gana en {(d.veredicto == 'K3 gana').sum()}, empata en {(d.veredicto == 'empate').sum()} y pierde en "
           f"{(d.veredicto == 'K3 pierde').sum()} de {m}; F1 mayor en {(d.f1_k3 > d.f1_xgb).sum()} de {m}.\n"
           f"Wilcoxon sobre las {m} diferencias de AUC: estadístico {w.statistic:.1f}, p = {w.pvalue:.3f}\n")
print(texto)
(AQUI / "publicos_otif_resumen.txt").write_text(texto, encoding="utf-8")
d.to_csv(AQUI / "publicos_otif_resumen.csv", index=False)
