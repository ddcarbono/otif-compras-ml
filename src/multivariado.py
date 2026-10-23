"""Exploración multivariada de la página EDA: PCA sobre las numéricas principales de la muestra.

Las variables pasan por la misma preparación que en los modelos (signo · log(1 + |x|), mediana en los vacíos y
estandarización), para que ninguna pese por su escala. PCA resume las 12 en pocos ejes."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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

