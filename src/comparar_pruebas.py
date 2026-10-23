"""DeLong, delta de Cliff e intervalos BCa para la mejor combinación de cada modelo en los 8 bloques externos.

Lee ``resultados/puntajes/<bloque>.csv.gz`` (los escribe ``src.puntajes`` en srvgdai) y escribe en
``resultados/pruebas/``:

- ``delong.csv``: XGBoost contra cada rival en cada bloque (48 pruebas), con p ajustado por Holm sobre las 48;
- ``resumen.csv``: por rival, bloques que gana XGBoost, bloques con DeLong significativo, diferencia media de AUC con su
  intervalo BCa 95 % y delta de Cliff sobre los 8 AUC por bloque;
- ``auc_bca.csv``: AUC medio de cada modelo en los 8 bloques con su intervalo BCa 95 %;
- ``validacion_delong.csv``: la prueba de un bloque real (2025-S2, XGBoost contra la logística) hecha aquí y con
  ``pROC::roc.test`` (``src/validar_delong.R``), para mostrar que coinciden también con 42.000 filas.

El bootstrap remuestrea órdenes dentro de cada bloque (B = 1000), porque las posiciones de una orden no son
independientes; DeLong, en cambio, supone filas independientes. La prueba 2026 no se usa.
Uso: ``python -m src.comparar_pruebas`` (unos 15 minutos; R solo para la validación).
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import datos, pruebas, validacion

ENTRADA = datos.RAIZ / "resultados" / "puntajes"
SALIDA = datos.RAIZ / "resultados" / "pruebas"
RSCRIPT = r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe"
MEJOR = "xgboost"


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    blo = {b: pd.read_csv(ENTRADA / f"{b}.csv.gz") for b in validacion.BLOQUES}
    modelos = [c for c in blo[validacion.BLOQUES[0]].columns if c not in ("y", "grupo")]
    rivales = [m for m in modelos if m != MEJOR]

    # DeLong por bloque, Holm sobre todas
    filas = []
    for b, d in blo.items():
        for r in rivales:
            filas.append({"bloque": b, "rival": r, **pruebas.delong(d.y, d[MEJOR], d[r])})
    dl = pd.DataFrame(filas)
    dl["p_holm"] = multipletests(dl.p, method="holm")[1]
    dl.to_csv(SALIDA / "delong.csv", index=False)

    # BCa: un solo bootstrap por órdenes dentro de cada bloque para los 7 AUC medios y las 6 diferencias
    def estadistico(muestras):
        a = np.array([[pruebas.auc(m.y.to_numpy(), m[c].to_numpy()) for c in modelos] for m in muestras]).mean(0)
        return np.concatenate([a, a[modelos.index(MEJOR)] - np.delete(a, modelos.index(MEJOR))])
    est, inf, sup = pruebas.bca([(d.grupo.to_numpy(), d) for d in blo.values()], estadistico, B=1000)
    k = len(modelos)
    pd.DataFrame({"modelo": modelos, "AUC": est[:k], "inf": inf[:k], "sup": sup[:k]}).sort_values(
        "AUC", ascending=False).to_csv(SALIDA / "auc_bca.csv", index=False)

    por_bloque = pd.DataFrame({m: [pruebas.auc(d.y, d[m]) for d in blo.values()] for m in modelos},
                              index=list(blo))
    res = []
    for j, r in enumerate([m for m in modelos if m != MEJOR]):
        g = dl[dl.rival == r]
        delta, mag = pruebas.cliff(por_bloque[MEJOR], por_bloque[r])
        res.append({"rival": r, "bloques_gana": int((por_bloque[MEJOR] > por_bloque[r]).sum()),
                    "bloques_delong_holm": int((g.p_holm < 0.05).sum()), "dif_AUC": est[k + j],
                    "dif_inf": inf[k + j], "dif_sup": sup[k + j], "cliff": delta, "magnitud": mag})
    pd.DataFrame(res).sort_values("dif_AUC").to_csv(SALIDA / "resumen.csv", index=False)

    # validación en un bloque real contra pROC
    d = blo["2025-S2"]
    with tempfile.TemporaryDirectory() as tmp:
        caso, sal = Path(tmp) / "caso.csv", Path(tmp) / "proc.csv"
        d.assign(s1=d[MEJOR], s2=d["logistica"])[["y", "s1", "s2"]].to_csv(caso, index=False)
        subprocess.run([RSCRIPT, str(datos.RAIZ / "src" / "validar_delong.R"), str(caso), str(sal)], check=True)
        r = pd.read_csv(sal).iloc[0]
    o = pruebas.delong(d.y, d[MEJOR], d["logistica"])
    pd.DataFrame([{"fuente": "src.pruebas", **{c: o[c] for c in ("AUC_1", "AUC_2", "z", "p")}},
                  {"fuente": "pROC::roc.test", "AUC_1": r.AUC_1, "AUC_2": r.AUC_2, "z": r.z, "p": r.p}]
                 ).to_csv(SALIDA / "validacion_delong.csv", index=False)
    print(pd.read_csv(SALIDA / "resumen.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
