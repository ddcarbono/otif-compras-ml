"""Cálculos del análisis exploratorio. Cada función recibe un DataFrame (el entrenamiento, salvo que se diga otra cosa)
y devuelve una tabla; las figuras las arma ``plots.py`` con esas tablas. Las usan el capítulo 2 del libro y
``preparar.py`` (datos del tablero), así que una cifra es la misma en los dos."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .tiempo import autocorrelacion, descomponer, estacionariedad, huecos, serie  # noqa: F401 (se usan desde aquí)
from .datos import CATEGORICAS, ENTIDADES, GRUPO, NOMBRE, NUMERICAS, OBJETIVO, SEMILLA, codificar_fuera_de_pliegue


# ---------------------------------------------------------------------------------------------------------- tamaño
def tamano(df):
    """Filas, órdenes, entidades independientes, tasa del objetivo y n/p, por partición y en total."""
    filas = []
    for nombre, d in [("total", df)] + list(df.groupby("particion", sort=False)):
        filas.append({"partición": nombre, "filas": len(d), "órdenes": d.oc_id.nunique(),
                      "proveedores": d.proveedor.nunique(), "materiales": d.material.nunique(),
                      "tardías": int(d[OBJETIVO].sum()), "tasa de tarde": d[OBJETIVO].mean()})
    t = pd.DataFrame(filas).set_index("partición")
    p = len(NUMERICAS) + len(CATEGORICAS) + len(ENTIDADES)
    t["n/p"] = t["filas"] / p
    return t


# ---------------------------------------------------------------------------------------------------------- calidad
def faltantes(df):
    """Porcentaje de vacíos por variable y, como evidencia del mecanismo, la tasa de tarde con y sin el dato.
    Si la tasa cambia mucho entre los dos, el vacío no es al azar (no es MCAR)."""
    filas = []
    for c in NUMERICAS + CATEGORICAS + ENTIDADES:
        v = df[c].isna()
        if v.any():
            filas.append({"variable": c, "grupo": GRUPO[c], "vacíos (%)": 100 * v.mean(),
                          "tasa con dato": df.loc[~v, OBJETIVO].mean(), "tasa sin dato": df.loc[v, OBJETIVO].mean()})
    t = pd.DataFrame(filas)
    t["diferencia (pp)"] = 100 * (t["tasa sin dato"] - t["tasa con dato"])
    return t.sort_values("vacíos (%)", ascending=False).reset_index(drop=True)


def vacios_por_bloque(df):
    """Los vacíos estructurales: qué fracción de cada bloque falta según la marca tiene_*."""
    filas = []
    for marca, prefijo in [("tiene_solped", "sp_"), ("tiene_imputacion", "imp_"), ("tiene_material", "mc_")]:
        cols = [c for c in df.columns if c.startswith(prefijo) and c in GRUPO]
        for valor, d in df.groupby(marca):
            filas.append({"marca": marca, "valor": int(valor), "filas": len(d),
                          "vacíos del bloque (%)": 100 * d[cols].isna().mean().mean(), "tasa de tarde": d[OBJETIVO].mean()})
    return pd.DataFrame(filas)


def duplicados(df):
    """Duplicados exactos en predictoras y objetivo, y casi duplicados: misma OC, material, cantidad y fecha prometida
    (posiciones repetidas dentro de una orden)."""
    pred = NUMERICAS + CATEGORICAS + ENTIDADES
    exactos = df.duplicated(pred + [OBJETIVO]).sum()
    conflicto = df.duplicated(pred).sum() - exactos           # mismas predictoras, distinto objetivo
    casi = df.duplicated(["oc_id", "material", "cantidad", "fecha_prometida"]).sum()
    return pd.DataFrame({"chequeo": ["duplicados exactos (predictoras y objetivo)",
                                     "mismas predictoras y distinto objetivo",
                                     "casi duplicados (misma OC, material, cantidad y fecha prometida)"],
                         "filas": [exactos, conflicto, casi], "% de las filas": 100 * np.array([exactos, conflicto, casi]) / len(df)})


def imposibles(df):
    """Reglas de rango: cuántas filas las violan."""
    reglas = {"cantidad ≤ 0": df.cantidad <= 0, "valor en COP ≤ 0": df.valor_cop <= 0,
              "precio unitario ≤ 0": df.precio_unitario <= 0, "plazo prometido < 0 días": df.plazo_prometido_dias < 0,
              "liberación con días hábiles < 0": df.dias_habiles_liberacion < 0,
              "tasa previa fuera de [0, 1]": ~df.prov_tasa_tarde_previa.dropna().between(0, 1).reindex(df.index, fill_value=True),
              "fecha prometida antes del pedido": df.fecha_prometida < df.fecha_pedido,
              "plazo al crear > 10 años": df.plazo_al_crear_dias > 3650,
              "posiciones abiertas del proveedor < 0": df.prov_pos_abiertas < 0,
              "antigüedad del proveedor < 0": df.pv_antiguedad_dias < 0}
    return pd.DataFrame({"regla": list(reglas), "filas": [int(m.sum()) for m in reglas.values()],
                         "% de las filas": [100 * m.mean() for m in reglas.values()]})


# ---------------------------------------------------------------------------------------------------------- univariado
def univariado_numericas(df):
    """Una fila por numérica: estadísticos, asimetría, curtosis en exceso (0 para la normal) y atípicos por IQR."""
    filas = []
    for c in NUMERICAS:
        x = df[c].dropna().astype(float)
        q1, q3 = x.quantile([.25, .75])
        iqr = q3 - q1
        filas.append({"variable": c, "nombre": NOMBRE[c], "grupo": GRUPO[c], "vacíos (%)": 100 * df[c].isna().mean(),
                      "distintos": x.nunique(), "media": x.mean(), "mediana": x.median(), "desv.": x.std(),
                      "p1": x.quantile(.01), "p99": x.quantile(.99), "asimetría": x.skew(), "curtosis exc.": x.kurt(),
                      "atípicos IQR (%)": 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0})
    return pd.DataFrame(filas)


def univariado_categoricas(df, min_filas=30):
    """Una fila por categórica o entidad: cardinalidad, categoría más frecuente y categorías raras."""
    filas = []
    for c in CATEGORICAS + ENTIDADES:
        f = df[c].value_counts(dropna=True)
        filas.append({"variable": c, "nombre": NOMBRE[c], "grupo": GRUPO[c], "vacíos (%)": 100 * df[c].isna().mean(),
                      "categorías": len(f), "más frecuente": f.index[0] if len(f) else None,
                      "% de la más frecuente": 100 * f.iloc[0] / f.sum() if len(f) else np.nan,
                      f"raras (< {min_filas} filas)": int((f < min_filas).sum())})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------------------------------------- bivariado
def _cramer(tabla):
    chi2, p, gl, _ = stats.chi2_contingency(tabla, correction=False)
    n = tabla.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(tabla.shape) - 1))), chi2, p, gl


def asociacion(df, muestra_mi=50_000):
    """Asociación de cada predictora con el objetivo, en una sola tabla.

    - Numéricas: AUC univariado (0,5 = no separa; se reporta max(AUC, 1 − AUC)), Mann-Whitney y su p, e información
      mutua (capta relaciones no monótonas), sobre una muestra de ``muestra_mi`` filas.
    - Categóricas: chi-cuadrado y V de Cramér.
    - Entidades: AUC de su codificación por objetivo fuera de pliegue (con miles de valores la V de Cramér se infla).
    Los p se ajustan por Holm sobre todas las pruebas."""
    y = df[OBJETIVO].astype(int)
    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(df), min(muestra_mi, len(df)), replace=False)
    filas = []
    num = df[NUMERICAS].astype(float)
    mi = mutual_info_classif(num.iloc[idx].fillna(num.median()), y.iloc[idx], random_state=SEMILLA)
    for c, m in zip(NUMERICAS, mi):
        x = num[c]
        ok = x.notna()
        auc = roc_auc_score(y[ok], x[ok]) if x[ok].nunique() > 1 else 0.5
        p = stats.mannwhitneyu(x[ok & (y == 1)], x[ok & (y == 0)]).pvalue if x[ok].nunique() > 1 else 1.0
        filas.append({"variable": c, "tipo": "numérica", "AUC": max(auc, 1 - auc), "dirección": "+" if auc >= .5 else "−",
                      "información mutua": m, "V de Cramér": np.nan, "p": p})
    for c in CATEGORICAS:
        tabla = pd.crosstab(df[c].fillna("(vacío)"), y)
        v, _, p, _ = _cramer(tabla)
        tasa = df.groupby(df[c].fillna("(vacío)"))[OBJETIVO].transform("mean")
        filas.append({"variable": c, "tipo": "categórica", "AUC": roc_auc_score(y, tasa), "dirección": "",
                      "información mutua": np.nan, "V de Cramér": v, "p": p})
    te = codificar_fuera_de_pliegue(df, y, df["oc_id"], ENTIDADES)
    for c in ENTIDADES:
        auc = roc_auc_score(y, te[f"te_{c}"])
        filas.append({"variable": c, "tipo": "entidad", "AUC": auc, "dirección": "", "información mutua": np.nan,
                      "V de Cramér": np.nan, "p": stats.mannwhitneyu(te.loc[y == 1, f"te_{c}"], te.loc[y == 0, f"te_{c}"]).pvalue})
    t = pd.DataFrame(filas)
    t["p Holm"] = multipletests(t["p"].clip(lower=1e-300), method="holm")[1]
    t.insert(1, "nombre", t.variable.map(NOMBRE))
    t.insert(2, "grupo", t.variable.map(GRUPO))
    return t.sort_values("AUC", ascending=False).reset_index(drop=True)


def contrastes_grupo(df, grupos=("linea", "clase_compras")):
    """Los contrastes por grupo: chi-cuadrado del retraso (con V de Cramér) y Kruskal-Wallis de los días de retraso
    (con épsilon², la fracción de la variación de rangos que explica el grupo)."""
    filas = []
    for g in grupos:
        v, chi2, p, gl = _cramer(pd.crosstab(df[g], df[OBJETIVO]))
        filas.append({"grupo": g, "prueba": "chi-cuadrado de y_tarde", "estadístico": chi2, "gl": gl, "p": p,
                      "efecto": v, "medida del efecto": "V de Cramér"})
        muestras = [d.dropna().to_numpy() for _, d in df.groupby(g)["dias_retraso"]]
        h, p = stats.kruskal(*muestras)
        n, k = sum(map(len, muestras)), len(muestras)
        filas.append({"grupo": g, "prueba": "Kruskal-Wallis de dias_retraso", "estadístico": h, "gl": k - 1, "p": p,
                      "efecto": (h - k + 1) / (n - k), "medida del efecto": "épsilon²"})
    t = pd.DataFrame(filas)
    t["p Holm"] = multipletests(t["p"].clip(lower=1e-300), method="holm")[1]
    return t


def tasa_por_grupo(df, g):
    """Filas, tasa de tarde y mediana de días de retraso por valor de g."""
    t = df.groupby(g).agg(filas=(OBJETIVO, "size"), tasa=(OBJETIVO, "mean"), mediana_retraso=("dias_retraso", "median"))
    return t.sort_values("filas", ascending=False).reset_index()


def spearman(df, variables):
    return df[variables].astype(float).corr(method="spearman")


def vif(df, variables, muestra=50_000):
    """Factor de inflación de la varianza: cuánto se infla la varianza de un coeficiente por la correlación de esa
    variable con las demás (1 = nada; más de 10 = colinealidad seria). Sobre una muestra, con la mediana en los vacíos."""
    x = df[variables].astype(float).sample(min(muestra, len(df)), random_state=SEMILLA)
    x = (x.fillna(x.median()) - x.mean()) / x.std()
    x = x.loc[:, x.std() > 0].assign(_const=1.0)
    return pd.DataFrame({"variable": x.columns[:-1],
                         "VIF": [variance_inflation_factor(x.to_numpy(), i) for i in range(x.shape[1] - 1)]}) \
        .sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------- multivariado
def _matriz_numerica(df, muestra):
    x = df[NUMERICAS].astype(float).sample(min(muestra, len(df)), random_state=SEMILLA)
    x = x.loc[:, x.nunique() > 1]
    # transformación simétrica de las colas largas (signo · log1p|x|) para que PCA no la dominen tres valores extremos
    asim = x.skew().abs() > 2
    x.loc[:, asim] = np.sign(x.loc[:, asim]) * np.log1p(x.loc[:, asim].abs())
    z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x))
    return x, z


def pca(df, muestra=50_000):
    """Varianza explicada por componente y cargas de los dos primeros."""
    x, z = _matriz_numerica(df, muestra)
    p = PCA(random_state=SEMILLA).fit(z)
    var = pd.DataFrame({"componente": np.arange(1, len(p.explained_variance_ratio_) + 1),
                        "varianza": p.explained_variance_ratio_, "acumulada": p.explained_variance_ratio_.cumsum()})
    cargas = pd.DataFrame(p.components_[:2].T, index=x.columns, columns=["CP1", "CP2"])
    return var, cargas


def atipicos_multivariados(df, muestra=50_000):
    """Isolation Forest sobre las numéricas: qué fracción marca como atípica y qué tasa de tarde tiene esa fracción."""
    x, z = _matriz_numerica(df, muestra)
    marca = IsolationForest(random_state=SEMILLA, n_jobs=-1).fit_predict(z) == -1
    y = df.loc[x.index, OBJETIVO]
    return pd.DataFrame({"grupo": ["atípicas", "resto"], "filas": [marca.sum(), (~marca).sum()],
                         "tasa de tarde": [y[marca].mean(), y[~marca].mean()]})


def grupos_kmedias(df, muestra=20_000, componentes=10, ks=range(2, 9)):
    """k-medias sobre los primeros componentes principales: silueta por k y, para el mejor, tamaño y tasa por grupo."""
    x, z = _matriz_numerica(df, muestra)
    zc = PCA(n_components=componentes, random_state=SEMILLA).fit_transform(z)
    sil = []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=5, random_state=SEMILLA).fit_predict(zc)
        sil.append({"k": k, "silueta": silhouette_score(zc, lab, sample_size=5000, random_state=SEMILLA)})
    sil = pd.DataFrame(sil)
    k = int(sil.loc[sil.silueta.idxmax(), "k"])
    lab = KMeans(n_clusters=k, n_init=5, random_state=SEMILLA).fit_predict(zc)
    y = df.loc[x.index, OBJETIVO].to_numpy()
    grupos = pd.DataFrame({"grupo": lab, "y": y}).groupby("grupo").y.agg(filas="size", tasa="mean").reset_index()
    return sil, grupos


# ---------------------------------------------------------------------------------------------------------- tiempo
def psi(ref, act, bins=10):
    """Índice de estabilidad de la población entre dos periodos (por deciles de la referencia):
    PSI = Σ (a − r) · ln(a / r). Menos de 0,1 estable; 0,1 a 0,25 cambio moderado; más de 0,25 cambio fuerte."""
    ref, act = pd.Series(ref).dropna().astype(float), pd.Series(act).dropna().astype(float)
    if ref.nunique() <= bins:                       # binarias y discretas: una clase por valor
        r = ref.value_counts(normalize=True)
        a = act.value_counts(normalize=True).reindex(r.index, fill_value=0) + 1e-6
        return float(np.sum((a - r - 1e-6) * np.log(a / (r + 1e-6))))
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:
        return np.nan
    cortes[0], cortes[-1] = -np.inf, np.inf
    r = np.histogram(ref, cortes)[0] / len(ref) + 1e-6
    a = np.histogram(act, cortes)[0] / len(act) + 1e-6
    return float(np.sum((a - r) * np.log(a / r)))


def deriva(df, variables):
    """PSI de cada variable del entrenamiento (2021-2024) frente a la validación (2025) y la prueba (2026). Solo mira
    distribuciones de las predictoras: no usa el objetivo de 2025 ni de 2026."""
    ent = df[df.particion == "entrenamiento"]
    filas = []
    for c in variables:
        filas.append({"variable": c, "nombre": NOMBRE[c],
                      "PSI 2025": psi(ent[c], df.loc[df.particion == "validacion", c]),
                      "PSI 2026": psi(ent[c], df.loc[df.particion == "prueba", c])})
    return pd.DataFrame(filas)


def entidades_nuevas(df):
    """Qué fracción de las filas de cada partición trae un valor de entidad que no aparece en el entrenamiento."""
    ent = df[df.particion == "entrenamiento"]
    filas = []
    for c in ENTIDADES:
        vistos = set(ent[c].dropna())
        fila = {"variable": c, "nombre": NOMBRE[c]}
        for p in ["validacion", "prueba"]:
            fila[f"nuevo en {p} (%)"] = 100 * (~df.loc[df.particion == p, c].isin(vistos)).mean()
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("nuevo en prueba (%)", ascending=False).reset_index(drop=True)


def oc_entre_particiones(df):
    """Una orden no debe quedar partida entre dos particiones."""
    return int((df.groupby("oc_id").particion.nunique() > 1).sum())
