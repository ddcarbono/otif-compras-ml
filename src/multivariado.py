"""Exploración multivariada de la página EDA: PCA y DBSCAN sobre las numéricas principales de la muestra.

Las variables pasan por la misma preparación que en los modelos (signo · log(1 + |x|), mediana en los vacíos y
estandarización), para que ninguna pese por su escala. PCA resume las 12 en pocos ejes; DBSCAN busca grupos densos en
esos ejes y deja como ruido lo que no cae en ninguno, sin fijar de antemano cuántos grupos hay."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def estandarizar(df, columnas):
    """Matriz estandarizada de ``columnas``: signo · log(1 + |x|), mediana en los vacíos y media 0, desviación 1."""
    x = df[columnas].astype(float)
    x = np.sign(x) * np.log1p(x.abs())
    x = x.fillna(x.median())
    sd = x.std().replace(0, 1)
    return ((x - x.mean()) / sd).to_numpy()


def pca(df, columnas):
    """(varianza explicada por componente, puntajes de cada fila, cargas de cada variable en las dos primeras)."""
    p = PCA(random_state=2026).fit(estandarizar(df, columnas))
    puntajes = p.transform(estandarizar(df, columnas))
    cargas = pd.DataFrame(p.components_[:2].T, index=columnas, columns=["PC1", "PC2"])
    return p.explained_variance_ratio_, puntajes, cargas


def eps_sugerido(puntos, vecinos):
    """El codo de la curva de distancias al k-ésimo vecino, aproximado por su percentil 90: la regla usual para eps."""
    d = NearestNeighbors(n_neighbors=vecinos).fit(puntos).kneighbors(puntos)[0][:, -1]
    return float(np.percentile(d, 90))


def dbscan(puntos, eps, vecinos):
    """Etiqueta de grupo de cada punto (-1 = ruido)."""
    return DBSCAN(eps=eps, min_samples=vecinos).fit_predict(puntos)


def resumen_grupos(etiquetas, y):
    """Tamaño y tasa de tarde de cada grupo, con el ruido aparte."""
    t = pd.DataFrame({"grupo": etiquetas, "y": np.asarray(y)}).groupby("grupo").y.agg(["size", "mean"]).reset_index()
    t["nombre"] = np.where(t.grupo < 0, "ruido", "grupo " + (t.grupo + 1).astype(str))
    return t.rename(columns={"size": "posiciones", "mean": "tasa"}).sort_values("posiciones", ascending=False)
