"""Boosting de árboles por histograma, escrito solo con numpy (el Entregable 3 pide la implementación propia del modelo
original sin bibliotecas de aprendizaje). Es la pieza que usa BOREAL para la parte f(x, c) y para el modelo log-normal del
plazo.

Qué hace, en el orden en que lo hace:

1. **Discretización** (`Discretizador`). Cada numérica se corta en a lo sumo 255 tramos por cuantiles del entrenamiento;
   el tramo 0 queda para el vacío. Cada categórica recibe un tramo por categoría vista (las de menos filas se juntan en
   el último tramo) y el 0 para vacía o nueva. Se guarda como una matriz uint8 de variables × filas.
2. **Gradiente y curvatura** de la pérdida en cada fila: logística (g = p − y, h = p(1 − p)) o cuadrática (g = ŷ − y,
   h = 1). El modelo arranca de un desplazamiento por fila (`init`) o del promedio.
3. **Un árbol por iteración, creciendo por hojas** (la de mayor ganancia primero, hasta `hojas`). En cada hoja se suman g,
   h y el conteo por tramo (histograma); el de la hija mayor sale de restar al de la madre el de la menor. Para cada
   variable se busca el corte que más baja la pérdida de segundo orden

       ganancia = S(G_I) + S(G_D) − S(G_I + G_D),  con  S(G) = −(2 T(G) w + (H + λ) w²),  w = −T(G) / (H + λ)

   donde T(G) = signo(G)·máx(|G| − α, 0) es el umbral L1 y λ el L2. En las numéricas el vacío se prueba a la izquierda y
   a la derecha. En las categóricas se ordenan las categorías por G / (H + suavizado) y se prueba cada prefijo (como el
   camino de muchos contra muchos de LightGBM, Fisher 1958); con cuatro o menos, una contra el resto.
4. **Monotonía**: una variable puede pedir que la salida baje cuando sube (−1). El corte en ella exige w_I ≥ w_D y las
   hijas heredan una cota en el punto medio, de modo que todo el árbol queda monótono en esa variable.
5. **Submuestreo**: filas (`submuestra`) y variables (`columnas`) al azar en cada árbol, con la semilla.
6. La salida de cada hoja se multiplica por la tasa de aprendizaje y se suma a la predicción de todas las filas.

Complejidad: discretizar O(n p log n); cada árbol O(n p' log L) para los histogramas (p' variables del árbol, L hojas,
gracias a la resta) más O(L p' B) para buscar cortes (B = 256 tramos); predecir O(n · profundidad) por árbol.
"""
import heapq

import numpy as np

VACIO = 0
MAX_TRAMOS = 255
UNO_CONTRA_RESTO = 4        # con tantas categorías o menos, una contra el resto
MAX_CATEGORIAS_CORTE = 32   # tope de categorías en el lado izquierdo de un corte categórico
L2_CATEGORICO = 10.0        # L2 extra en los cortes categóricos (como cat_l2 de LightGBM)


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Discretizador:
    """Tramos por cuantiles para las numéricas y por categoría para las categóricas."""

    def __init__(self, max_tramos=MAX_TRAMOS, min_categoria=1):
        self.max_tramos, self.min_categoria = max_tramos, min_categoria

    def fit(self, columnas, categoricas):
        """columnas: lista de arreglos 1-D (una por variable); categoricas: lista de bool."""
        self.categoricas = list(categoricas)
        self.bordes, self.categorias = [], []
        for x, es_cat in zip(columnas, self.categoricas):
            if es_cat:
                x = np.asarray(x, dtype=object)
                vista = np.array([v for v in x if v is not None and v == v], dtype=object)
                valores, conteo = np.unique(vista.astype(str), return_counts=True)
                orden = np.argsort(-conteo, kind="stable")
                self.categorias.append(dict((v, i + 1) for i, v in enumerate(valores[orden][:self.max_tramos])))
                self.bordes.append(None)
            else:
                x = np.asarray(x, dtype=float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    b = np.array([0.0])
                else:
                    u = np.unique(x)
                    if len(u) <= self.max_tramos:
                        b = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([u[0]])
                    else:
                        b = np.unique(np.quantile(x, np.linspace(0, 1, self.max_tramos + 1)[1:-1]))
                self.bordes.append(b)
                self.categorias.append(None)
        return self

    def transform(self, columnas):
        n = len(columnas[0])
        T = np.zeros((len(columnas), n), dtype=np.uint8)
        for j, x in enumerate(columnas):
            if self.categoricas[j]:
                mapa = self.categorias[j]
                x = np.asarray(x, dtype=object)
                T[j] = [mapa.get(str(v), VACIO) if (v is not None and v == v) else VACIO for v in x]
            else:
                x = np.asarray(x, dtype=float)
                t = np.searchsorted(self.bordes[j], x, side="left") + 1
                t[~np.isfinite(x)] = VACIO
                T[j] = np.minimum(t, 255)
        return T


class _Arbol:
    """Arreglos de nodos: variable, corte (tramo), vacío a la izquierda, categórico, máscara de categorías, hijos, valor
    y cobertura (filas del submuestreo que pasaron por el nodo, la que usa TreeSHAP). Las máscaras de los cortes
    categóricos se guardan empaquetadas (32 bytes por nodo categórico)."""

    def __init__(self):
        self.var, self.corte, self.vacio_izq, self.es_cat, self.mascara = [], [], [], [], []
        self.izq, self.der, self.valor, self.cobertura = [], [], [], []

    def nuevo(self, valor, cobertura):
        for lista, v in ((self.var, -1), (self.corte, 0), (self.vacio_izq, False), (self.es_cat, False),
                         (self.mascara, None), (self.izq, -1), (self.der, -1), (self.valor, valor),
                         (self.cobertura, cobertura)):
            lista.append(v)
        return len(self.valor) - 1

    def cerrar(self):
        self.var = np.array(self.var, dtype=np.int32)
        self.corte = np.array(self.corte, dtype=np.int32)
        self.vacio_izq = np.array(self.vacio_izq, dtype=bool)
        self.es_cat = np.array(self.es_cat, dtype=bool)
        self.cat_idx = np.full(len(self.valor), -1, dtype=np.int32)
        bits = []
        for i, s in enumerate(self.mascara):
            if s is not None:
                self.cat_idx[i] = len(bits)
                bits.append(np.packbits(s))
        self.bits = np.array(bits, dtype=np.uint8).reshape(-1, 32)
        del self.mascara
        self.izq = np.array(self.izq, dtype=np.int32)
        self.der = np.array(self.der, dtype=np.int32)
        self.valor = np.array(self.valor, dtype=float)
        self.cobertura = np.array(self.cobertura, dtype=float)
        return self

    def _mascaras(self):
        if getattr(self, "_m", None) is None:
            self._m = (np.unpackbits(self.bits, axis=1).astype(bool) if len(self.bits)
                       else np.zeros((1, 256), dtype=bool))
        return self._m

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_m", None)
        return d

    def izquierda(self, k, t):
        """Si las filas con tramo t van a la izquierda en los nodos k."""
        m = self._mascaras()
        return np.where(self.es_cat[k], m[np.maximum(self.cat_idx[k], 0), t],
                        np.where(t == VACIO, self.vacio_izq[k], t <= self.corte[k]))

    def hoja(self, T, filas=None):
        """Índice de la hoja de cada fila (T: variables × filas, uint8)."""
        n = T.shape[1] if filas is None else len(filas)
        cols = np.arange(T.shape[1]) if filas is None else filas
        nodo = np.zeros(n, dtype=np.int32)
        activo = np.flatnonzero(self.izq[nodo] >= 0)
        while len(activo):
            k = nodo[activo]
            izq = self.izquierda(k, T[self.var[k], cols[activo]])
            nodo[activo] = np.where(izq, self.izq[k], self.der[k])
            activo = activo[self.izq[nodo[activo]] >= 0]
        return nodo

    def predecir(self, T, filas=None):
        return self.valor[self.hoja(T, filas)]

    def caminos(self):
        """Por cada hoja: su índice y la lista de (nodo, va a la izquierda) desde la raíz."""
        out, pila = [], [(0, [])]
        while pila:
            nodo, camino = pila.pop()
            if self.izq[nodo] < 0:
                out.append((nodo, camino))
            else:
                pila.append((int(self.izq[nodo]), camino + [(nodo, True)]))
                pila.append((int(self.der[nodo]), camino + [(nodo, False)]))
        return out


def _umbral_l1(G, alfa):
    return np.sign(G) * np.maximum(np.abs(G) - alfa, 0.0)


def _salida(G, H, lam, alfa, lo, hi):
    return np.clip(-_umbral_l1(G, alfa) / (H + lam), lo, hi)


def _puntaje(G, H, lam, alfa, w):
    return -(2.0 * _umbral_l1(G, alfa) * w + (H + lam) * w * w)


class BoostingHistograma:
    """Boosting de árboles por histograma (pérdida logística o cuadrática).

    Parámetros con los nombres de LightGBM que reemplaza: learning_rate, num_leaves, min_child_samples, subsample,
    colsample_bytree, reg_lambda, reg_alpha, cat_smooth, min_data_per_group; además n_estimators, objetivo
    ("binaria" o "cuadratica"), monotonia (lista de 0 y −1 por variable) y semilla."""

    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0, cat_smooth=10.0, min_data_per_group=100,
                 min_sum_hessian=1e-3, objetivo="binaria", monotonia=None, semilla=2026, **_ignorados):
        self.n_estimators, self.learning_rate, self.num_leaves = int(n_estimators), learning_rate, int(num_leaves)
        self.min_child_samples, self.subsample, self.colsample_bytree = int(min_child_samples), subsample, colsample_bytree
        self.reg_lambda, self.reg_alpha, self.cat_smooth = reg_lambda, reg_alpha, cat_smooth
        self.min_data_per_group, self.min_sum_hessian = int(min_data_per_group), min_sum_hessian
        self.objetivo, self.monotonia, self.semilla = objetivo, monotonia, semilla

    # ---- histogramas y cortes -----------------------------------------------------------------------------------------
    @staticmethod
    def _histograma(T, vars_, filas, g, h):
        gf, hf = g[filas], h[filas]
        G = np.empty((len(vars_), 256))
        H = np.empty((len(vars_), 256))
        C = np.empty((len(vars_), 256))
        for i, j in enumerate(vars_):
            t = T[j, filas]
            G[i] = np.bincount(t, gf, 256)
            H[i] = np.bincount(t, hf, 256)
            C[i] = np.bincount(t, minlength=256)
        return G, H, C

    def _mejor_corte(self, G, H, C, vars_, lo, hi):
        """Mejor corte de un nodo: (ganancia, i_var, corte, vacio_izq, mascara, wI, wD) o None."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._mejor_corte_(G, H, C, vars_, lo, hi)

    def _mejor_corte_(self, G, H, C, vars_, lo, hi):
        lam, alfa, mc, msh = self.reg_lambda, self.reg_alpha, self.min_child_samples, self.min_sum_hessian
        Gt, Ht = G[0].sum(), H[0].sum()
        w0 = _salida(Gt, Ht, lam, alfa, lo, hi)
        padre = _puntaje(Gt, Ht, lam, alfa, w0)
        mejor = None
        num = ~self._cat[vars_]
        if num.any():
            iv = np.flatnonzero(num)
            g0, h0, c0 = G[iv, 0:1], H[iv, 0:1], C[iv, 0:1]
            GL, HL, CL = np.cumsum(G[iv, 1:], 1), np.cumsum(H[iv, 1:], 1), np.cumsum(C[iv, 1:], 1)
            for vac_izq in (False, True):
                gl, hl, cl = (GL + g0, HL + h0, CL + c0) if vac_izq else (GL, HL, CL)
                gr, hr, cr = Gt - gl, Ht - hl, C[iv].sum(1, keepdims=True) - cl
                ok = (cl >= mc) & (cr >= mc) & (hl >= msh) & (hr >= msh)
                wl = _salida(gl, hl, lam, alfa, lo, hi)
                wr = _salida(gr, hr, lam, alfa, lo, hi)
                mono = self._mono[vars_[iv]][:, None]
                ok &= ~((mono < 0) & (wl < wr)) & ~((mono > 0) & (wl > wr))
                gan = np.where(ok, _puntaje(gl, hl, lam, alfa, wl) + _puntaje(gr, hr, lam, alfa, wr) - padre, -np.inf)
                k = int(np.argmax(gan))
                a, b = divmod(k, gan.shape[1])
                if gan[a, b] > 1e-12 and (mejor is None or gan[a, b] > mejor[0]):
                    mejor = (float(gan[a, b]), int(iv[a]), b + 1, vac_izq, None, float(wl[a, b]), float(wr[a, b]))
        for i in np.flatnonzero(~num):
            cand = self._corte_categorico(G[i], H[i], C[i], Gt, Ht, padre, lo, hi)
            if cand is not None and (mejor is None or cand[0] > mejor[0]):
                mejor = (cand[0], int(i), 0, False, cand[1], cand[2], cand[3])
        return mejor

    def _corte_categorico(self, g, h, c, Gt, Ht, padre, lo, hi):
        lam, alfa = self.reg_lambda + L2_CATEGORICO, self.reg_alpha
        Ct = c.sum()
        usados = np.flatnonzero(c[1:] > 0) + 1
        if len(usados) < 2:
            return None
        if len(usados) <= UNO_CONTRA_RESTO:
            candidatos = [[u] for u in usados]
        else:
            usados = usados[c[usados] >= self.cat_smooth]
            if len(usados) < 2:
                return None
            orden = usados[np.argsort(g[usados] / (h[usados] + self.cat_smooth), kind="stable")]
            tope = min(MAX_CATEGORIAS_CORTE, (len(orden) + 1) // 2)
            candidatos = [orden[:k] for k in range(1, tope + 1)] + [orden[::-1][:k] for k in range(1, tope + 1)]
        mejor = None
        for s in candidatos:
            gl, hl, cl = g[s].sum(), h[s].sum(), c[s].sum()
            gr, hr, cr = Gt - gl, Ht - hl, Ct - cl
            if min(cl, cr) < max(self.min_child_samples, self.min_data_per_group if len(s) > 1 else 0):
                continue
            if min(hl, hr) < self.min_sum_hessian:
                continue
            wl, wr = _salida(gl, hl, lam, alfa, lo, hi), _salida(gr, hr, lam, alfa, lo, hi)
            gan = _puntaje(gl, hl, lam, alfa, wl) + _puntaje(gr, hr, lam, alfa, wr) - padre
            if gan > 1e-12 and (mejor is None or gan > mejor[0]):
                m = np.zeros(256, dtype=bool)
                m[s] = True
                mejor = (float(gan), m, float(wl), float(wr))
        return mejor

    # ---- un árbol ----------------------------------------------------------------------------------------------------
    def _arbol(self, T, g, h, filas, vars_):
        arbol = _Arbol()
        Gt, Ht = g[filas].sum(), h[filas].sum()
        raiz = arbol.nuevo(float(_salida(Gt, Ht, self.reg_lambda, self.reg_alpha, -np.inf, np.inf)), len(filas))
        hist = {raiz: self._histograma(T, vars_, filas, g, h)}
        miembros = {raiz: filas}
        cotas = {raiz: (-np.inf, np.inf)}
        cola, hojas = [], 1
        corte = self._mejor_corte(*hist[raiz], vars_, -np.inf, np.inf)
        if corte is not None:
            heapq.heappush(cola, (-corte[0], raiz, corte))
        while cola and hojas < self.num_leaves:
            _, nodo, (gan, i, cte, vac_izq, mascara, wl, wr) = heapq.heappop(cola)
            j = vars_[i]
            f = miembros.pop(nodo)
            t = T[j, f]
            izq = mascara[t] if mascara is not None else np.where(t == VACIO, vac_izq, t <= cte)
            fi, fd = f[izq], f[~izq]
            lo, hi = cotas.pop(nodo)
            ci, cd = (lo, hi), (lo, hi)
            if self._mono[j] != 0:
                medio = (wl + wr) / 2
                ci, cd = ((medio, hi), (lo, medio)) if self._mono[j] < 0 else ((lo, medio), (medio, hi))
            a, b = arbol.nuevo(wl, len(fi)), arbol.nuevo(wr, len(fd))
            arbol.var[nodo], arbol.corte[nodo], arbol.vacio_izq[nodo] = j, cte, vac_izq
            arbol.es_cat[nodo], arbol.mascara[nodo], arbol.izq[nodo], arbol.der[nodo] = mascara is not None, mascara, a, b
            H_p = hist.pop(nodo)
            chica, grande = (a, b) if len(fi) <= len(fd) else (b, a)
            hist[chica] = self._histograma(T, vars_, fi if chica == a else fd, g, h)
            hist[grande] = tuple(P - Q for P, Q in zip(H_p, hist[chica]))
            miembros[a], miembros[b], cotas[a], cotas[b] = fi, fd, ci, cd
            hojas += 1
            for hijo in (a, b):
                c = self._mejor_corte(*hist[hijo], vars_, *cotas[hijo])
                if c is not None:
                    heapq.heappush(cola, (-c[0], hijo, c))
                else:
                    hist.pop(hijo)
        return arbol.cerrar(), miembros

    # ---- ajuste y predicción -------------------------------------------------------------------------------------------
    def _grad(self, y, z):
        if self.objetivo == "binaria":
            p = sigmoide(z)
            return p - y, np.maximum(p * (1 - p), 1e-16)
        return z - y, np.ones_like(z)

    def fit(self, T, y, categoricas, init=None, peso=None):
        """T: variables × filas (uint8, de un Discretizador); categoricas: bool por variable; init: logit de partida por fila."""
        y = np.asarray(y, dtype=float)
        p_, n = T.shape
        self._cat = np.asarray(categoricas, dtype=bool)
        self._mono = np.zeros(p_, dtype=np.int8) if self.monotonia is None else np.asarray(self.monotonia, dtype=np.int8)
        if init is None:
            m = y.mean()
            self.base_ = float(np.log(m / (1 - m))) if self.objetivo == "binaria" else float(m)
            z = np.full(n, self.base_)
        else:
            self.base_ = 0.0
            z = np.asarray(init, dtype=float).copy()
        rng = np.random.default_rng(self.semilla)
        self.arboles_ = []
        k = max(1, int(round(self.colsample_bytree * p_)))
        todas = np.arange(n)
        fuera = np.ones(n, dtype=bool)
        for _ in range(self.n_estimators):
            g, h = self._grad(y, z)
            if peso is not None:
                g, h = g * peso, h * peso
            filas = todas if self.subsample >= 1 else np.flatnonzero(rng.random(n) < self.subsample)
            vars_ = np.sort(rng.choice(p_, k, replace=False)) if k < p_ else np.arange(p_)
            a, miembros = self._arbol(T, g, h, filas, vars_)
            a.valor = a.valor * self.learning_rate
            self.arboles_.append(a)
            # las filas del submuestreo ya saben su hoja; solo las de fuera recorren el árbol
            for hoja, f in miembros.items():
                z[f] += a.valor[hoja]
            if len(filas) < n:
                fuera[:] = True
                fuera[filas] = False
                resto = np.flatnonzero(fuera)
                z[resto] += a.predecir(T, resto)
        return self

    def predict_raw(self, T, n_arboles=None):
        z = np.full(T.shape[1], self.base_)
        for a in self.arboles_[:n_arboles]:
            z += a.predecir(T)
        return z

    # ---- TreeSHAP ------------------------------------------------------------------------------------------------------
    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_shap", None)
        return d

    def _estructura_shap(self):
        """Los caminos de todas las hojas de todos los árboles, agrupados por número de variables distintas.

        Cada camino guarda el valor de la hoja, sus variables distintas, la fracción de cobertura de cada una (el
        producto de cobertura del hijo / cobertura del nodo en los cortes de esa variable) y las condiciones (nodo,
        lado) que una fila debe cumplir para seguirlo."""
        if getattr(self, "_shap", None) is not None:
            return self._shap
        desfase, grupos, cond, n_elem, esperado = 0, {}, [], 0, 0.0
        for a in self.arboles_:
            caminos = a.caminos()
            esperado += float(sum(a.valor[h] * a.cobertura[h] for h, _ in caminos) / a.cobertura[0])
            for hoja, camino in caminos:
                por_var = {}
                for nodo, izq in camino:
                    hijo = a.izq[nodo] if izq else a.der[nodo]
                    e = por_var.setdefault(int(a.var[nodo]), [1.0, []])
                    e[0] *= a.cobertura[hijo] / a.cobertura[nodo]
                    e[1].append((desfase + nodo, izq))
                d = len(por_var)
                if d == 0:
                    continue
                g = grupos.setdefault(d, {"v": [], "f": [], "z": [], "e": []})
                g["v"].append(a.valor[hoja])
                fs, zs, es = [], [], []
                for f, (z, conds) in por_var.items():
                    fs.append(f)
                    zs.append(z)
                    es.append(n_elem)
                    cond.extend((n_elem, nodo, izq) for nodo, izq in conds)
                    n_elem += 1
                g["f"].append(fs)
                g["z"].append(zs)
                g["e"].append(es)
            desfase += len(a.valor)
        grupos = {d: {k: np.array(v) for k, v in g.items()} for d, g in grupos.items()}
        c = np.array(cond, dtype=np.int64).reshape(-1, 3)
        self._shap = dict(grupos=grupos, cond_e=c[:, 0], cond_n=c[:, 1], cond_izq=c[:, 2].astype(bool),
                          n_elem=n_elem, esperado=esperado + self.base_)
        return self._shap

    def shap(self, T, fila_lote=64):
        """Valores SHAP exactos de árbol (Lundberg et al. 2020, algoritmo 2, con la dependencia del camino) de cada
        fila: arreglo filas × variables y el valor esperado; predict_raw = esperado + suma de la fila.

        Se calcula por camino de hoja y no por recursión: el peso de cada subconjunto es el polinomio que arma EXTEND,
        que no depende del orden de las variables, así que se arma de una vez con las variables distintas del camino y
        se vectoriza sobre caminos y filas (la misma idea de GPUTreeShap, Mitchell et al. 2022)."""
        s = self._estructura_shap()
        n_var, n = T.shape
        out = np.zeros((n, n_var))
        for ini in range(0, n, fila_lote):
            sub = T[:, ini:ini + fila_lote]
            r = sub.shape[1]
            lados = []
            for a in self.arboles_:
                lado = np.zeros((len(a.valor), r), dtype=bool)
                ki = np.flatnonzero(a.izq >= 0)
                if len(ki):
                    lado[ki] = a.izquierda(ki[:, None], sub[a.var[ki]])
                lados.append(lado)
            lados = np.vstack(lados)
            falla = np.zeros((s["n_elem"], r))
            np.add.at(falla, s["cond_e"], (lados[s["cond_n"]] != s["cond_izq"][:, None]).astype(float))
            O_todo = (falla == 0).astype(float)
            fi = np.zeros((n_var, r))
            for d, g in s["grupos"].items():
                Z = g["z"][:, :, None]                       # caminos × d × 1
                O = O_todo[g["e"]]                           # caminos × d × filas
                P = len(g["v"])
                W = np.zeros((P, d + 1, r))
                W[:, 0] = 1.0                                # elemento ficticio de la raíz (z = o = 1)
                for u in range(1, d + 1):                    # EXTEND con el elemento u
                    z, o = Z[:, u - 1], O[:, u - 1]
                    for i in range(u - 1, -1, -1):
                        W[:, i + 1] += o * W[:, i] * (i + 1) / (u + 1)
                        W[:, i] = z * W[:, i] * (u - i) / (u + 1)
                total = np.zeros((P, d, r))                  # UNWOUND de cada elemento a la vez (eje 1)
                cero = np.zeros((P, d, r))
                sig = np.repeat(W[:, d:d + 1], d, axis=1)
                o_seguro = np.where(O > 0, O, 1.0)
                for i in range(d - 1, -1, -1):
                    tmp = sig / ((i + 1) * o_seguro)
                    total += tmp
                    sig = W[:, i:i + 1] - tmp * Z * (d - i)
                    cero += W[:, i:i + 1] / (Z * (d - i))
                peso = np.where(O > 0, total, cero) * (d + 1)
                aporte = peso * (O - Z) * g["v"][:, None, None]
                np.add.at(fi, g["f"].ravel(), aporte.reshape(P * d, r))
            out[ini:ini + r] = fi.T
        return out, s["esperado"]
