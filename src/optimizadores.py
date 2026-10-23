"""Los cuatro optimizadores de hiperparámetros, con el mismo presupuesto de E evaluaciones del objetivo.

Un espacio es una lista de dimensiones ``(nombre, tipo, extremos, log)``: ``tipo`` es ``"int"``, ``"float"`` o
``"cat"`` (con ``extremos`` = las opciones). Todos los optimizadores reciben ``objetivo(params) -> float`` (el AUC en la
validación interna, a maximizar) y devuelven ``(mejores_params, historial)``; el historial tiene una fila por evaluación
con su valor y el tiempo de reloj acumulado, para las curvas de mejor valor contra evaluaciones y contra tiempo.

- **Grid**: una rejilla regular con a lo sumo E puntos. Las categóricas entran con todas sus opciones y las numéricas
  reparten el resto del presupuesto (niveles centrados en el cubo unitario, en escala log cuando corresponde). Con seis
  dimensiones y E = 20 la rejilla queda en dos niveles por dimensión: es el defecto conocido de la búsqueda en rejilla.
- **Random**: E puntos uniformes en el cubo unitario, decodificados al espacio (Bergstra y Bengio, 2012).
- **Optuna (TPE)**: estimador de Parzen en árbol. Modela por separado la densidad de los puntos buenos, l(x), y la de los
  malos, g(x) (el corte es el cuantil γ de los valores observados), y propone el x que maximiza l(x)/g(x), que equivale a
  maximizar la mejora esperada (Bergstra et al., 2011). Arranca con min(10, E/2) puntos al azar; ``multivariate=True``
  modela las dimensiones juntas.
- **DEAP (genético)**: individuos en [0, 1]^d; selección por torneo de 3, cruce uniforme (probabilidad 0,7, cada gen
  0,5), mutación gaussiana (probabilidad 0,3, σ = 0,15, cada gen 0,3) recortada a [0, 1], elitismo de 1. Población de
  round(1,4·√E) (6 con E = 20). Un individuo que decodifica a parámetros ya evaluados no gasta presupuesto; la búsqueda
  se detiene en la evaluación E aunque sea a mitad de generación. Se registra la diversidad (desviación media de los
  genes) por generación.
"""
import math
import time

import numpy as np

OPTIMIZADORES = ["grid", "random", "optuna", "deap"]


def decodificar(espacio, u):
    """Un punto del cubo unitario a parámetros del modelo."""
    p = {}
    for (nombre, tipo, ext, log), ui in zip(espacio, u):
        ui = float(min(max(ui, 0.0), 1.0))
        if tipo == "cat":
            p[nombre] = ext[min(int(ui * len(ext)), len(ext) - 1)]
            continue
        a, b = ext
        v = math.exp(math.log(a) + ui * (math.log(b) - math.log(a))) if log else a + ui * (b - a)
        p[nombre] = int(round(v)) if tipo == "int" else float(v)
    return p


def _clave(p):
    return tuple(sorted((k, float(f"{v:.10g}") if isinstance(v, float) else v) for k, v in p.items()))


class _Registro:
    """Evalúa, guarda el historial y no repite parámetros ya evaluados."""

    def __init__(self, objetivo, presupuesto):
        self.objetivo, self.E, self.t0 = objetivo, presupuesto, time.time()
        self.hist, self.vistos = [], {}

    @property
    def agotado(self):
        return len(self.hist) >= self.E

    def __call__(self, p, **extra):
        k = _clave(p)
        if k in self.vistos:
            return self.vistos[k]
        v = float(self.objetivo(p))
        if not np.isfinite(v):
            v = 0.0
        self.vistos[k] = v
        self.hist.append({"eval": len(self.hist) + 1, "valor": v, "segundos": time.time() - self.t0, "params": p, **extra})
        return v

    def mejor(self):
        h = max(self.hist, key=lambda r: r["valor"])
        return h["params"], self.hist


def grid(espacio, E, objetivo, semilla):
    cats = [i for i, d in enumerate(espacio) if d[1] == "cat"]
    nums = [i for i, d in enumerate(espacio) if d[1] != "cat"]
    n_cat = math.prod(len(espacio[i][2]) for i in cats) if cats else 1
    resto = max(E // n_cat, 1)
    k = [1] * len(nums)
    sube = True
    while sube:  # un nivel más en cada dimensión, por turnos, mientras el producto quepa en el presupuesto
        sube = False
        for i in range(len(nums)):
            if math.prod(k) // k[i] * (k[i] + 1) <= resto:
                k[i] += 1
                sube = True
    niveles = {}
    for i, ki in zip(nums, k):
        niveles[i] = [(m + 0.5) / ki for m in range(ki)]
    for i in cats:
        n = len(espacio[i][2])
        niveles[i] = [(m + 0.5) / n for m in range(n)]
    puntos = [[]]
    for i in range(len(espacio)):
        puntos = [p + [v] for p in puntos for v in niveles[i]]
    r = _Registro(objetivo, E)
    for u in puntos[:E]:
        r(decodificar(espacio, u))
    return r.mejor()


def aleatoria(espacio, E, objetivo, semilla):
    rng = np.random.default_rng(semilla)
    r = _Registro(objetivo, E)
    intentos = 0
    while not r.agotado and intentos < 50 * E:
        r(decodificar(espacio, rng.random(len(espacio))))
        intentos += 1
    return r.mejor()


def tpe(espacio, E, objetivo, semilla):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    r = _Registro(objetivo, E)

    def f(trial):
        p = {}
        for nombre, tipo, ext, log in espacio:
            if tipo == "cat":
                p[nombre] = trial.suggest_categorical(nombre, list(ext))
            elif tipo == "int":
                p[nombre] = trial.suggest_int(nombre, ext[0], ext[1], log=log)
            else:
                p[nombre] = trial.suggest_float(nombre, ext[0], ext[1], log=log)
        return r(p)

    s = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=min(10, E // 2), multivariate=True)
    estudio = optuna.create_study(direction="maximize", sampler=s)
    while not r.agotado and len(estudio.trials) < 5 * E:   # los repetidos no gastan presupuesto
        estudio.optimize(f, n_trials=1)
    return r.mejor()


def genetico(espacio, E, objetivo, semilla):
    import random

    from deap import base, creator, tools
    if not hasattr(creator, "Aptitud"):
        creator.create("Aptitud", base.Fitness, weights=(1.0,))
        creator.create("Individuo", list, fitness=creator.Aptitud)
    random.seed(semilla)
    d = len(espacio)
    tb = base.Toolbox()
    tb.register("individuo", tools.initRepeat, creator.Individuo, random.random, d)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("cruzar", tools.cxUniform, indpb=0.5)
    tb.register("mutar", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.3)
    tb.register("seleccionar", tools.selTournament, tournsize=3)
    P_CX, P_MUT = 0.7, 0.3
    tam = max(4, round(1.4 * math.sqrt(E)))
    r = _Registro(objetivo, E)
    diversidad = []

    def evaluar(pob, gen):
        for ind in pob:
            if not ind.fitness.valid:
                if r.agotado and _clave(decodificar(espacio, ind)) not in r.vistos:
                    return False
                ind.fitness.values = (r(decodificar(espacio, ind), generacion=gen),)
        return True

    pob = tb.poblacion(n=tam)
    gen = 0
    seguir = evaluar(pob, gen)
    diversidad.append(float(np.std(np.array(pob), axis=0).mean()))
    while seguir and not r.agotado and gen < 10 * E:
        gen += 1
        elite = tools.selBest(pob, 1)
        hijos = [tb.clone(i) for i in tb.seleccionar(pob, len(pob) - 1)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if random.random() < P_CX:
                tb.cruzar(a, b)
                del a.fitness.values, b.fitness.values
        for h in hijos:
            if random.random() < P_MUT:
                tb.mutar(h)
                h[:] = [min(max(g, 0.0), 1.0) for g in h]
                del h.fitness.values
        seguir = evaluar(hijos, gen)
        pob = elite + [h for h in hijos if h.fitness.valid]
        diversidad.append(float(np.std(np.array(pob), axis=0).mean()))
    p, hist = r.mejor()
    for fila in hist:
        fila["diversidad_generacion"] = diversidad[min(fila.get("generacion", 0), len(diversidad) - 1)]
    return p, hist


FUNCIONES = {"grid": grid, "random": aleatoria, "optuna": tpe, "deap": genetico}


def optimizar(nombre, espacio, E, objetivo, semilla):
    return FUNCIONES[nombre](espacio, E, objetivo, semilla)
