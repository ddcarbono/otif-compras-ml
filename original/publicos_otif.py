"""Tres conjuntos publicos con el mismo protocolo (Entregable 3, bloque D; Daniel, 2026-10-23): el XGBoost ganador de los
112 contra el algoritmo propio (K3: boosting con efectos aleatorios cruzados por entidad, correccion C, reduccion ordinal
cuando hay una medida continua del desenlace y efectos ordenados en el tiempo cuando hay orden temporal).

Conjuntos (OpenML, descargados a datos/publicos/, que no viaja):
  airlines  Airlines_DepDelay_1M (id 42721, CC0, Bureau of Transportation Statistics). Positivo: salida con mas de 15 min
            de retraso (la definicion de la BTS). Medida continua: DepDelay en minutos, cortes 0, 15, 30 y 60. Entidades:
            aerolinea, origen, destino y ruta. El archivo no trae el anio (mezcla varios): la particion es por mes del
            anio (entrena enero-agosto, valida septiembre-octubre, prueba noviembre-diciembre), estacional y no
            estrictamente temporal; los efectos ordenados usan el mes. Muestra aleatoria de 100.000 filas.
  amazon    Amazon_employee_access (id 4135, publica, Kaggle 2013). Positivo: acceso negado. Entidades: RESOURCE, MGR_ID,
            ROLE_TITLE y ROLE_DEPTNAME; los demas niveles de rol entran codificados por objetivo. Sin tiempo ni medida
            continua: particion estratificada 60/20/20 con la semilla.
  click     Click_prediction_small v5 (id 1220, publica, KDD Cup 2012). Positivo: clic. Entidades: ad_id, advertiser_id,
            query_id y user_id; keyword, title, description y url por objetivo. Sin tiempo: particion estratificada 60/20/20.

Protocolo, igual para los dos modelos y los tres conjuntos:
  - Preprocesamiento comun: las columnas de muchos valores van con codificacion por objetivo suavizada (M = 20), fuera de
    pliegue en el entrenamiento; las numericas tal cual. K3 cambia la codificacion de sus cuatro entidades por efectos
    aleatorios, como en OTIF.
  - Mismo presupuesto: Optuna (TPE) con E = 20 evaluaciones sobre el AUC de validacion, misma semilla. XGBoost con el
    espacio de los 112 (src/experimentos.py); K3 con el de su boosting (tasa, hojas, minimo por hoja, columnas, alfa, arboles).
  - Se reajusta con entrenamiento + validacion y se mide una vez en la prueba: AUC con DeLong pareado (Sun y Xu 2014),
    intervalo de la diferencia por bootstrap (200), F1 al umbral de maximo F1 de la validacion.
El modelo log-normal y las variables nuevas de OTIF no aplican: necesitan un plazo prometido y la historia de entregas.

Segunda ronda (elegida antes de correr; ver las funciones nyc_bus, zurich, diabetes y noshow):
  python -I publicos_otif.py <datos>/publicos --solo nyc_bus,zurich,diabetes,noshow --sufijo ronda2

Uso (ml_venv): python -I publicos_otif.py <datos>/publicos [--ensayos 20] [--solo airlines,amazon,click] [--sufijo x]
Deja publicos_otif.txt y publicos_otif.csv aqui.
"""
import io
import re
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))  # se corre con -I (lee datos descargados): el directorio del script no entra solo

import numpy as np
import optuna
import pandas as pd
from scipy.io import arff
from scipy.stats import norm
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier

import modelo_original_otif as mo

sys.stdout.reconfigure(encoding="utf-8")  # -I ignora PYTHONIOENCODING y la consola de Windows no imprime «−»

DATOS = Path(sys.argv[1])
arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
ENSAYOS, SOLO = arg("--ensayos", 20), arg("--solo", "airlines,amazon,click").split(",")
SUFIJO = arg("--sufijo", "")
SEMILLA, M = 2026, 20
optuna.logging.set_verbosity(optuna.logging.WARNING)
salida, filas = [], []
t0 = time.time()
reloj = lambda: f"{(time.time() - t0) / 60:.1f} min"


def anotar(texto):
    print(texto, flush=True)
    salida.append(texto)


def leer_arff(ruta):
    d, _ = arff.loadarff(ruta)
    df = pd.DataFrame(d)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8")
    return df


# --- conjuntos: (df, y, dias o None, cortes, corte objetivo, entidades RE, otras categoricas, numericas, periodo o None, particion)
def airlines():
    df = leer_arff(DATOS / "airlines" / "airlines_depdelay_1m.arff")
    df = df[df.DepDelay.notna()].sample(100_000, random_state=SEMILLA).reset_index(drop=True)
    df["ruta"] = df.Origin + "-" + df.Dest
    df["periodo"] = df.Month.astype(int)
    part = np.select([df.Month <= 8, df.Month <= 10], ["ent", "val"], "pru")
    return dict(df=df, y=(df.DepDelay > 15).astype(int).to_numpy(), dias=df.DepDelay.to_numpy(float), cortes=[0, 15, 30, 60],
                corte=15, re=["UniqueCarrier", "Origin", "Dest", "ruta"], cat=[],
                num=["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "Distance"], periodo="periodo", part=part)


def estratificada(y):
    idx = np.arange(len(y))
    ent, resto = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEMILLA)
    val, pru = train_test_split(resto, test_size=0.5, stratify=y[resto], random_state=SEMILLA)
    part = np.empty(len(y), dtype=object)
    part[ent], part[val], part[pru] = "ent", "val", "pru"
    return part


def amazon():
    df = leer_arff(DATOS / "amazon" / "amazon_employee_access.arff")
    y = (df.target.astype(int) == 0).astype(int).to_numpy()
    return dict(df=df, y=y, dias=None, cortes=[0], corte=0, re=["RESOURCE", "MGR_ID", "ROLE_TITLE", "ROLE_DEPTNAME"],
                cat=["ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"], num=[], periodo=None,
                part=estratificada(y))


def click():
    df = leer_arff(DATOS / "click" / "click_prediction_small.arff")
    y = df.click.astype(int).to_numpy()
    for c in ["ad_id", "advertiser_id", "query_id", "user_id", "keyword_id", "title_id", "description_id", "url_hash"]:
        df[c] = df[c].astype("int64").astype(str)
    return dict(df=df, y=y, dias=None, cortes=[0], corte=0, re=["ad_id", "advertiser_id", "query_id", "user_id"],
                cat=["keyword_id", "title_id", "description_id", "url_hash"], num=["impression", "depth", "position"],
                periodo=None, part=estratificada(y))


# --- segunda ronda (2026-10-23): elegidos ANTES de correr con el criterio de la hipotesis que dejo la primera: la
# propuesta sirve con entidades repetidas, orden en el tiempo y desenlace continuo u ordinal. Entran los que cumplen al
# menos dos de los tres, y se reportan todos.
def leer_arff_texto(ruta):
    """ARFF con atributos de texto, que scipy no lee: encabezado a mano y los datos como CSV con comillas simples."""
    cols, lineas = [], open(ruta, encoding="utf-8", errors="replace").read().split("\n")
    for i, l in enumerate(lineas):
        if l.lower().startswith("@attribute"):
            cols.append(re.match(r"@attribute\s+('[^']*'|\"[^\"]*\"|\S+)", l, re.I).group(1).strip("'\""))
        if l.lower().startswith("@data"):
            break
    return pd.read_csv(io.StringIO("\n".join(lineas[i + 1:])), header=None, names=cols, quotechar="'", escapechar="\\",
                       na_values=["?"], skipinitialspace=True, low_memory=False)


def temporal(orden, n_periodos):
    """Particion 60/20/20 por el orden temporal y periodos para los efectos ordenados (cuantiles del orden)."""
    r = pd.Series(orden).rank(method="first").to_numpy() / len(orden)
    part = np.select([r <= 0.6, r <= 0.8], ["ent", "val"], "pru")
    return part, np.minimum((r * n_periodos).astype(int), n_periodos - 1)


def minutos(texto):
    """'20 MINS', '20-30 min', '1 hour', '1HR 30MIN' -> minutos (el primer numero; horas por 60). Vacio si no hay numero."""
    if not isinstance(texto, str):
        return np.nan
    t = texto.lower()
    h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", t)
    m = re.search(r"(\d+)", t)
    if h:
        extra = re.search(r"(?:h|hr|hrs|hour|hours)\D*(\d+)\s*m", t)
        return float(h.group(1)) * 60 + (float(extra.group(1)) if extra else 0)
    return float(m.group(1)) if m else np.nan


def nyc_bus():
    df = leer_arff_texto(DATOS / "nyc_bus" / "bus_breakdown_delays_nyc.arff")
    df["min"] = df.How_Long_Delayed.map(minutos)
    df["fecha"] = pd.to_datetime(df.Occurred_On, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df[df["min"].between(0, 300) & df.fecha.notna()].reset_index(drop=True)
    df["hora"], df["dia_semana"], df["mes"] = df.fecha.dt.hour, df.fecha.dt.dayofweek, df.fecha.dt.month
    df["avisados"] = (df.Number_Of_Students_On_The_Bus.fillna(0)).astype(float)
    for c in ["Bus_Company_Name", "Route_Number", "Schools_Serviced", "Bus_No", "Reason", "Boro", "Run_Type",
              "School_Age_or_PreK", "Breakdown_or_Running_Late"]:
        df[c] = df[c].fillna("(vacío)").astype(str)
    part, per = temporal(df.fecha.values.astype("int64"), 12)
    df["periodo"] = per
    return dict(df=df, y=(df["min"] > 30).astype(int).to_numpy(), dias=df["min"].to_numpy(float), cortes=[0, 15, 30, 45, 60],
                corte=30, re=["Bus_Company_Name", "Route_Number", "Schools_Serviced", "Bus_No"],
                cat=["Reason", "Boro", "Run_Type", "School_Age_or_PreK", "Breakdown_or_Running_Late"],
                num=["hora", "dia_semana", "mes", "avisados"], periodo="periodo", part=part)


def zurich():
    df = leer_arff(DATOS / "zurich" / "delays_zurich_transport.arff")
    df["linea_sentido"] = df.line_number + "-" + df.direction
    y = (df.delay > 180).astype(int).to_numpy()
    return dict(df=df, y=y, dias=df.delay.to_numpy(float), cortes=[0, 60, 180, 300], corte=180,
                re=["line_number", "stop_id", "linea_sentido"], cat=["vehicle_type", "weekday", "direction"],
                num=["time", "temp", "windspeed_max", "windspeed_avg", "precipitation", "dew_point", "humidity"],
                periodo=None, part=estratificada(y))


def diabetes():
    df = leer_arff_texto(DATOS / "diabetes" / "diabetes130us.arff")
    nivel = df.readmitted.map({"NO": 0, ">30": 1, "<30": 2}).to_numpy(float)
    cat = [c for c in df.columns if df[c].dtype == object and c not in ("readmitted", "medical_specialty", "diag_1", "diag_2")]
    for c in cat + ["medical_specialty", "diag_1", "diag_2"]:
        df[c] = df[c].fillna("(vacío)").astype(str)
    df["paciente"] = df.patient_nbr.astype("int64").astype(str)
    num = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures",
           "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    part, per = temporal(df.encounter_id.to_numpy(), 10)
    df["periodo"] = per
    return dict(df=df, y=(nivel == 2).astype(int), dias=nivel, cortes=[0, 1], corte=1,
                re=["paciente", "medical_specialty", "diag_1", "diag_2"], cat=cat, num=num, periodo="periodo", part=part)


def noshow():
    df = leer_arff_texto(DATOS / "noshow" / "medical_appointment_no_shows.arff")
    agenda, cita = pd.to_datetime(df.ScheduledDay).dt.tz_localize(None), pd.to_datetime(df.AppointmentDay).dt.tz_localize(None)
    df["espera_dias"] = (cita.dt.normalize() - agenda.dt.normalize()).dt.days
    df = df[df.espera_dias >= 0].reset_index(drop=True)
    cita = pd.to_datetime(df.AppointmentDay).dt.tz_localize(None)
    df["dia_semana_cita"], df["hora_agenda"] = cita.dt.dayofweek, pd.to_datetime(df.ScheduledDay).dt.hour
    df["paciente"] = df.PatientId.astype("int64").astype(str)
    df["Neighbourhood"], df["Gender"] = df.Neighbourhood.astype(str), df.Gender.astype(str)
    part, per = temporal(cita.values.astype("int64"), 6)
    df["periodo"] = per
    return dict(df=df, y=(df["No-show"] == "Yes").astype(int).to_numpy(), dias=None, cortes=[0], corte=0,
                re=["paciente", "Neighbourhood"], cat=["Gender"],
                num=["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "espera_dias",
                     "dia_semana_cita", "hora_agenda"], periodo="periodo", part=part)


# --- preprocesamiento comun --------------------------------------------------------------------------------------------
def te(col_e, y_e, col_a):
    t = pd.DataFrame({"k": col_e.values, "y": y_e}).groupby("k").y.agg(["sum", "count"])
    g = y_e.mean()
    return ((col_a.map(t["sum"]).fillna(0) + M * g) / (col_a.map(t["count"]).fillna(0) + M)).to_numpy()


def matrices(c, ent, apl, con_re):
    """X de entrenamiento (te fuera de pliegue) y de aplicacion; con_re = True deja fuera las te de las entidades de K3."""
    df, y = c["df"], c["y"]
    cols_te = c["cat"] + ([] if con_re else c["re"])
    Xe = df.loc[ent, c["num"]].reset_index(drop=True).astype(float)
    Xa = df.loc[apl, c["num"]].reset_index(drop=True).astype(float)
    ie = np.flatnonzero(ent)
    for col in cols_te:
        oof = np.zeros(len(ie))
        for a, b in KFold(5, shuffle=True, random_state=SEMILLA).split(ie):
            oof[b] = te(df[col].iloc[ie[a]].astype(str), y[ie[a]], df[col].iloc[ie[b]].astype(str))
        Xe[f"te_{col}"] = oof
        Xa[f"te_{col}"] = te(df[col][ent].astype(str), y[ent], df[col][apl].astype(str))
    return Xe, Xa


# --- modelos -----------------------------------------------------------------------------------------------------------
def xgb_logit(prm, Xe, ye, Xa):
    m = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEMILLA, eval_metric="logloss", **prm).fit(Xe, ye)
    return m.predict(Xa, output_margin=True)


def espacio_xgb(tr):
    return dict(n_estimators=tr.suggest_int("n_estimators", 100, 600), learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=tr.suggest_int("max_depth", 3, 10), subsample=tr.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=tr.suggest_float("colsample_bytree", 0.3, 1.0),
                min_child_weight=tr.suggest_float("min_child_weight", 1, 100, log=True))


def espacio_k3(tr):
    return dict(learning_rate=tr.suggest_float("learning_rate", 0.01, 0.1, log=True), num_leaves=tr.suggest_int("num_leaves", 15, 127),
                min_child_samples=tr.suggest_int("min_child_samples", 10, 300, log=True), subsample=0.9,
                colsample_bytree=tr.suggest_float("colsample_bytree", 0.3, 1.0), reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10, log=True),
                arboles=tr.suggest_int("arboles", 100, 1200))


class K3Rapido(mo.BoostingMixtoOrdinal):
    """K3 con los efectos ya estimados. Los efectos se estiman con los hiperparametros fijos de PRM_EFECTOS, que no
    dependen de los que busca Optuna: se calculan una vez por particion (efectos_k3) y cada ensayo solo ajusta los
    arboles, igual que la segunda mitad de BoostingMixtoOrdinal.fit."""

    def ajustar_arboles(self, X, dias, efectos, d_arboles):
        self.niveles, self.b, self.s2 = efectos["niveles"], efectos["b"], efectos["s2"]
        Xa, ya = self._ampliar(X, dias)
        prm = dict(self.prm)
        if self.monotono:
            prm["monotone_constraints"] = [0] * X.shape[1] + [-1]
        self.modelo = mo.lgb.LGBMClassifier(n_estimators=self.arboles, random_state=SEMILLA, verbose=-1, n_jobs=-1, **prm)
        self.modelo.fit(Xa, ya, init_score=np.tile(d_arboles, len(mo.CORTES)))
        self.arboles_usados = self.arboles
        return self


def efectos_k3(c, ent, Xe):
    """Efectos de las entidades con todo ent (los que se usan al predecir) y lo que ven los arboles: los ordenados por
    periodo si el conjunto tiene tiempo (K3), o los mismos con todo ent si no."""
    mo.ENTIDADES_RE, mo.CORTES, mo.CORTE_OBJETIVO = c["re"], c["cortes"], c["corte"]
    df = c["df"]
    ge = {e: df[e][ent].astype(str).reset_index(drop=True) for e in c["re"]}
    dias = c["dias"][ent] if c["dias"] is not None else c["y"][ent].astype(float)
    y_bin = (dias > c["corte"]).astype(int)
    # sin ordenes: un solo grupo, cuyo efecto la correccion C deja en 0 (un efecto por fila absorberia la etiqueta)
    oc = pd.Series(np.zeros(ent.sum(), dtype=int))
    m = mo.BoostingMixtoOrdinal(mo.PRM_EFECTOS, centrar=True)
    d = m._estimar_efectos(Xe, y_bin, ge, oc, np.zeros(ent.sum()))
    if c["periodo"]:
        per = df[c["periodo"]][ent].to_numpy()
        d = np.zeros(ent.sum())
        for t in sorted(set(per))[1:]:
            antes, este = per < t, per == t
            aux = mo.BoostingMixtoOrdinal(mo.PRM_EFECTOS, centrar=True)
            aux._estimar_efectos(Xe[antes], y_bin[antes], {e: s[antes] for e, s in ge.items()}, oc[antes], np.zeros(antes.sum()))
            d[este] = aux.desplazamiento({e: s[este] for e, s in ge.items()})
    anotar(f"   efectos de K3 estimados con {ent.sum()} filas; varianzas {({e: round(v, 3) for e, v in m.s2.items() if e in c['re']})} | {reloj()}")
    return dict(niveles=m.niveles, b=m.b, s2=m.s2, d_arboles=d, dias=dias)


def k3_logit(c, prm, ef, apl, Xe, Xa):
    prm = dict(prm)
    arboles = prm.pop("arboles")
    m = K3Rapido(prm, arboles=arboles, monotono=len(c["cortes"]) > 1, centrar=True)
    m.ajustar_arboles(Xe, ef["dias"], ef, ef["d_arboles"])
    ga = {e: c["df"][e][apl].astype(str).reset_index(drop=True) for e in c["re"]}
    return m.logit(Xa, ga, corte=c["corte"])


# --- DeLong pareado (Sun y Xu 2014) --------------------------------------------------------------------------------------
def _rangos_medios(x):
    o = np.argsort(x)
    z = x[o]
    r = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[o] = r
    return out


def delong(y, s1, s2):
    """AUC de cada uno, diferencia, su error estandar y p bilateral."""
    pos, neg = s1[y == 1], s1[y == 0]
    m, n = (y == 1).sum(), (y == 0).sum()
    v10, v01, aucs = [], [], []
    for s in (s1, s2):
        p_, n_ = s[y == 1], s[y == 0]
        tx, ty, tz = _rangos_medios(p_), _rangos_medios(n_), _rangos_medios(np.r_[p_, n_])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
        aucs.append(auc)
    sx, sy = np.cov(np.array(v10)), np.cov(np.array(v01))
    var = (sx[0, 0] + sx[1, 1] - 2 * sx[0, 1]) / m + (sy[0, 0] + sy[1, 1] - 2 * sy[0, 1]) / n
    d = aucs[0] - aucs[1]
    return aucs[0], aucs[1], d, np.sqrt(var), 2 * norm.sf(abs(d) / np.sqrt(var))


def umbral_f1(y, p):
    prec, sens, u = precision_recall_curve(y, p)
    return float(u[np.argmax(2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12))])


# --- corrida -------------------------------------------------------------------------------------------------------------
anotar(f"publicos_otif.py | E = {ENSAYOS} por modelo y conjunto | semilla {SEMILLA}")
for nombre in SOLO:
    c = {"airlines": airlines, "amazon": amazon, "click": click, "nyc_bus": nyc_bus, "zurich": zurich, "diabetes": diabetes,
         "noshow": noshow}[nombre]()
    y, part = c["y"], c["part"]
    ent, val, pru = part == "ent", part == "val", part == "pru"
    anotar(f"\n{nombre}: {len(y)} filas | tasa del positivo {y.mean():.3f} | entrenamiento {ent.sum()}, validación {val.sum()}, "
           f"prueba {pru.sum()} (tasa {y[pru].mean():.3f}) | efectos de {c['re']} | cortes {c['cortes']} | periodo {c['periodo']}")
    res = {}
    for modelo, espacio in [("XGBoost", espacio_xgb), ("K3", espacio_k3)]:
        con_re = modelo == "K3"
        Xe, Xv = matrices(c, ent, val, con_re)
        ef = efectos_k3(c, ent, Xe) if con_re else None

        def objetivo(tr):
            prm = espacio(tr)
            z = xgb_logit(prm, Xe, y[ent], Xv) if modelo == "XGBoost" else k3_logit(c, prm, ef, val, Xe, Xv)
            tr.set_user_attr("z", z)
            return roc_auc_score(y[val], z)

        est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
        est.optimize(objetivo, n_trials=ENSAYOS)
        mejor = est.best_trial
        p_val = 1 / (1 + np.exp(-mejor.user_attrs["z"]))
        u = umbral_f1(y[val], p_val)
        # reajuste con entrenamiento + validacion y una sola medida en la prueba
        ev = ent | val
        Xev, Xp = matrices(c, ev, pru, con_re)
        ef = efectos_k3(c, ev, Xev) if con_re else None
        z = xgb_logit(mejor.params, Xev, y[ev], Xp) if modelo == "XGBoost" else k3_logit(c, mejor.params, ef, pru, Xev, Xp)
        res[modelo] = (z, u, mejor.value)
        anotar(f"   {modelo:8} AUC validación {mejor.value:.4f} (mejor de {ENSAYOS}) | prueba {roc_auc_score(y[pru], z):.4f} "
               f"| F1 {f1_score(y[pru], 1 / (1 + np.exp(-z)) >= u):.4f} al umbral {u:.3f} | {mejor.params} | {reloj()}")
    yp = y[pru]
    a_k, a_x, d, ee, p = delong(yp, res["K3"][0], res["XGBoost"][0])
    rng = np.random.default_rng(SEMILLA)
    boot = []
    for _ in range(200):
        i = rng.integers(0, len(yp), len(yp))
        if 0 < yp[i].sum() < len(i):
            boot.append(roc_auc_score(yp[i], res["K3"][0][i]) - roc_auc_score(yp[i], res["XGBoost"][0][i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    f1k = f1_score(yp, 1 / (1 + np.exp(-res["K3"][0])) >= res["K3"][1])
    f1x = f1_score(yp, 1 / (1 + np.exp(-res["XGBoost"][0])) >= res["XGBoost"][1])
    anotar(f"   K3 − XGBoost en la prueba: ΔAUC {d:+.4f} [{lo:+.4f}; {hi:+.4f}] | DeLong EE {ee:.4f}, p = {p:.4f} | ΔF1 {f1k - f1x:+.4f}")
    filas.append(dict(conjunto=nombre, n=len(y), tasa=round(y.mean(), 4), n_prueba=int(pru.sum()), auc_xgb=round(a_x, 4),
                      auc_k3=round(a_k, 4), dif=round(d, 4), ic_lo=round(lo, 4), ic_hi=round(hi, 4), p_delong=round(p, 5),
                      f1_xgb=round(f1x, 4), f1_k3=round(f1k, 4), auc_val_xgb=round(res["XGBoost"][2], 4), auc_val_k3=round(res["K3"][2], 4)))

suf = f"_{SUFIJO}" if SUFIJO else ""
pd.DataFrame(filas).to_csv(AQUI / f"publicos_otif{suf}.csv", index=False)
(AQUI / f"publicos_otif{suf}.txt").write_text("\n".join(salida) + "\n", encoding="utf-8")
