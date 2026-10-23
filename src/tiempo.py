"""Series de tiempo del EDA: volumen y tasa por periodo, huecos, STL, estacionariedad y autocorrelación. Va aparte
de ``eda.py`` porque el tablero la usa en vivo con los filtros, y así no carga scikit-learn."""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

OBJETIVO = "y_tarde"


def serie(df, freq="MS", por=None):
    """Volumen y tasa de tarde por periodo de fecha_pedido (y por grupo si se pide)."""
    por = [por] if isinstance(por, str) else list(por or [])
    claves = [pd.Grouper(key="fecha_pedido", freq=freq)] + por
    t = df.groupby(claves).agg(filas=(OBJETIVO, "size"), tardias=(OBJETIVO, "sum")).reset_index()
    t["tasa"] = t.tardias / t.filas
    return t


def huecos(df):
    """Cobertura de la fecha de pedido: días con y sin órdenes, y semanas sin ninguna."""
    dias = df.fecha_pedido.dt.normalize()
    rango = pd.date_range(dias.min(), dias.max(), freq="D")
    con = rango.isin(dias.unique())
    semanas = serie(df, "W-MON")
    return pd.DataFrame({"medida": ["primera fecha", "última fecha", "días del periodo", "días con órdenes",
                                    "días hábiles (lun-vie) sin órdenes", "semanas sin órdenes"],
                         "valor": [str(dias.min().date()), str(dias.max().date()), len(rango), int(con.sum()),
                                   int(((~con) & (rango.dayofweek < 5)).sum()), int((semanas.filas == 0).sum())]})


def descomponer(mensual, periodo=12):
    """STL de la tasa mensual: tendencia, estacionalidad y residuo."""
    s = mensual.set_index("fecha_pedido")["tasa"].asfreq("MS")
    r = STL(s, period=periodo, robust=True).fit()
    return pd.DataFrame({"fecha": s.index, "observada": s.values, "tendencia": r.trend.values,
                         "estacionalidad": r.seasonal.values, "residuo": r.resid.values})


def estacionariedad(s):
    """ADF (H0: raíz unitaria, no estacionaria) y KPSS (H0: estacionaria alrededor de una constante). Leídas juntas:
    ADF rechaza y KPSS no rechaza → estacionaria."""
    s = pd.Series(s).dropna()
    a = adfuller(s, autolag="AIC")
    k = kpss(s, regression="c", nlags="auto")
    return pd.DataFrame({"prueba": ["ADF", "KPSS"], "H0": ["raíz unitaria", "estacionaria"],
                         "estadístico": [a[0], k[0]], "p": [a[1], k[1]]})


def autocorrelacion(s, rezagos=18):
    s = pd.Series(s).dropna()
    n = min(rezagos, len(s) // 2 - 1)
    return pd.DataFrame({"rezago": np.arange(n + 1), "ACF": acf(s, nlags=n), "PACF": pacf(s, nlags=n),
                         "banda": 1.96 / np.sqrt(len(s))})
