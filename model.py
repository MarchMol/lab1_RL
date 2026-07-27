"""
MDP de logistica de drones urbanos - implementacion completa segun el diseno
de las Tasks 1 y 2 del notebook (sin las simplificaciones de la Task 3).

Estado s = (x, y, b): posicion en la grilla y bateria (Task 1 seccion 1).
Clima (w) y congestion (c) tambien forman parte del diseno del estado, pero
son variables estocasticas del entorno que aqui se fijan una unica vez por
celda al construir el ambiente (una de las 3 categorias, elegida al azar con
semilla fija) -- el agente solo consulta el valor ya asignado a su coordenada
via `weather_at`/`congestion_at`, no se vuelven a muestrear en cada paso.

Bateria: sistema simplificado de bins de 5% (0,5,...,100). Cualquier accion
que mantiene al dron en el aire (movimiento u Hover) consume 1 bin por paso;
`Cargar` en una estacion de carga llena la bateria a 100% en un unico paso
(Task 1 seccion 4, Transicion 3). Se puede fijar el nivel de bateria inicial
al construir el `GridWorld` (`battery_initial`) o al pedir un estado inicial
concreto via `GridWorld.initial_state`.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np

GRID_SIZE = 5
DEST = (4, 4)                                   # celda destino fija del ambiente
NO_FLY_ZONES = {(1, 3), (2, 2), (3, 1)}         # zonas de vuelo restringido
CHARGING_STATIONS = {(0, 0), (4, 0), (0, 4)}    # estaciones de carga

BATTERY_MAX = 100
BATTERY_BIN = 5
BATTERY_LEVELS: Tuple[int, ...] = tuple(range(0, BATTERY_MAX + 1, BATTERY_BIN))
BATTERY_CRITICAL = 10  # b < 10% -> Transicion 4 (Task 1 seccion 4)

SQRT2 = math.sqrt(2)

Weather = Literal["calma", "moderado", "fuerte"]
Congestion = Literal["libre", "moderado", "riesgoso"]
WEATHER_LEVELS: Tuple[Weather, ...] = ("calma", "moderado", "fuerte")
CONGESTION_LEVELS: Tuple[Congestion, ...] = ("libre", "moderado", "riesgoso")
WEATHER_SEVERITY: Dict[Weather, float] = {"calma": 0.0, "moderado": 1.0, "fuerte": 2.0}


class A(Enum):
    NORTE = (0, 1)
    SUR = (0, -1)
    ESTE = (1, 0)
    OESTE = (-1, 0)
    NE = (1, 1)
    NO = (-1, 1)
    SE = (1, -1)
    SO = (-1, -1)
    HOVER = "hover"
    CARGAR = "cargar"


MOVE_ACTIONS = (A.NORTE, A.SUR, A.ESTE, A.OESTE, A.NE, A.NO, A.SE, A.SO)
DIAGONAL_ACTIONS = (A.NE, A.NO, A.SE, A.SO)

# Para cada accion de movimiento: delta intencionado y los dos deltas de
# deriva lateral por viento (Task 1 seccion 4). Para diagonales, la deriva
# colapsa hacia las dos componentes cardinales de esa diagonal.
_MOVE_DELTAS: Dict[A, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {
    A.NORTE: ((0, 1), (-1, 1), (1, 1)),
    A.SUR:   ((0, -1), (-1, -1), (1, -1)),
    A.ESTE:  ((1, 0), (1, 1), (1, -1)),
    A.OESTE: ((-1, 0), (-1, 1), (-1, -1)),
    A.NE: ((1, 1), (0, 1), (1, 0)),
    A.NO: ((-1, 1), (0, 1), (-1, 0)),
    A.SE: ((1, -1), (0, -1), (1, 0)),
    A.SO: ((-1, -1), (0, -1), (-1, 0)),
}

ACTION_ARROWS: Dict[A, Tuple[int, int]] = {a: d[0] for a, d in _MOVE_DELTAS.items()}

# Costo de tiempo r_tiempo(a) (Task 1 seccion 3)
TIME_COST: Dict[A, float] = {a: (-SQRT2 if a in DIAGONAL_ACTIONS else -1.0) for a in MOVE_ACTIONS}
TIME_COST[A.HOVER] = -1.0
TIME_COST[A.CARGAR] = -10.0


@dataclass(frozen=True)
class S:
    x: int
    y: int
    b: int  # bateria, bin de 5% (0..100)

    def __repr__(self):
        return f"({self.x},{self.y},b={self.b})"


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def is_free(x: int, y: int) -> bool:
    return in_bounds(x, y) and (x, y) not in NO_FLY_ZONES


class GridWorld:
    """MDP tabular (S, A, p, r, gamma) para el GridWorld 5x5 de drones,
    fiel al diseno completo de la Task 1 (incluye bateria, clima y
    congestion, a diferencia de la version reducida de la Task 3)."""

    def __init__(self, gamma: float = 0.98, battery_initial: int = 100, seed: int = 42):
        self.gamma = gamma
        if battery_initial not in BATTERY_LEVELS:
            raise ValueError(f"battery_initial debe ser multiplo de {BATTERY_BIN} entre 0 y {BATTERY_MAX}")
        self.battery_initial = battery_initial

        # Clima y congestion: aleatorios pero fijos por celda (una unica
        # muestra al construir el ambiente, con semilla fija por defecto
        # para que el mapa sea reproducible entre corridas).
        rng = random.Random(seed)
        self._weather_map: Dict[Tuple[int, int], Weather] = {
            (x, y): rng.choice(WEATHER_LEVELS)
            for x in range(GRID_SIZE) for y in range(GRID_SIZE)
        }
        self._congestion_map: Dict[Tuple[int, int], Congestion] = {
            (x, y): rng.choice(CONGESTION_LEVELS)
            for x in range(GRID_SIZE) for y in range(GRID_SIZE)
        }

        self.states: List[S] = self._build_states()

    def weather_at(self, x: int, y: int) -> Weather:
        return self._weather_map[(x, y)]

    def congestion_at(self, x: int, y: int) -> Congestion:
        return self._congestion_map[(x, y)]

    # ---------- Espacio de estados ----------
    def _build_states(self) -> List[S]:
        states = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in NO_FLY_ZONES:
                    continue
                for b in BATTERY_LEVELS:
                    states.append(S(x, y, b))
        return states

    def initial_state(self, x: int, y: int, battery: int = None) -> S:
        """Estado inicial de un episodio, con nivel de bateria configurable
        (por defecto `self.battery_initial`)."""
        b = self.battery_initial if battery is None else battery
        if b not in BATTERY_LEVELS:
            raise ValueError(f"battery debe ser multiplo de {BATTERY_BIN} entre 0 y {BATTERY_MAX}")
        return S(x, y, b)

    def is_terminal(self, s: S) -> bool:
        # Los dos unicos estados terminales absorbentes del diseno (Task 1
        # seccion 4): llegar al destino (exito) o agotar la bateria (falla).
        return (s.x, s.y) == DEST or s.b == 0

    # ---------- Espacio de acciones (con enmascaramiento, Task 1 seccion 2) ----------
    def actions(self, s: S) -> List[A]:
        if self.is_terminal(s):
            return []
        valid = []
        for a in MOVE_ACTIONS:
            dx, dy = _MOVE_DELTAS[a][0]
            if is_free(s.x + dx, s.y + dy):
                valid.append(a)
        valid.append(A.HOVER)  # siempre disponible
        if (s.x, s.y) in CHARGING_STATIONS:
            valid.append(A.CARGAR)
        return valid

    # ---------- Probabilidades de riesgo segun clima/bateria (Task 1 seccion 4) ----------
    def _risk_probs(self, x: int, y: int, b: int) -> Tuple[float, float, float]:
        """Devuelve (p_intencionado, p_deriva_total, p_catastrofico) para una
        accion que mantiene al dron en vuelo, partiendo de (x,y) con bateria
        b. Bateria critica domina sobre el clima (Transicion 4); si no,
        depende del clima local (Transiciones 1 y 2). El caso `moderado` no
        esta definido explicitamente en la Task 1 y se interpola entre
        `calma` y `fuerte`."""
        if b < BATTERY_CRITICAL:
            return 0.70, 0.20, 0.10          # Transicion 4
        w = self.weather_at(x, y)
        if w == "fuerte":
            return 0.55, 0.40, 0.05          # Transicion 2
        if w == "moderado":
            return 0.75, 0.25, 0.0           # interpolacion calma/fuerte
        return 0.90, 0.10, 0.0               # Transicion 1 (calma)

    # ---------- Funcion de transicion p(s'|s,a) (Task 1 seccion 4) ----------
    def transition(self, s: S, a: A) -> Dict[S, float]:
        if a == A.CARGAR:
            return {S(s.x, s.y, BATTERY_MAX): 1.0}  # Transicion 3: determinista, llena a 100%

        b_after = max(0, s.b - BATTERY_BIN)
        p_intended, p_drift, p_cat = self._risk_probs(s.x, s.y, s.b)

        outcomes: Dict[S, float] = {}
        if p_cat > 0:
            # fallo catastrofico: el dron no completa la maniobra, se queda
            # en su celda actual pero con la bateria agotada
            outcomes[S(s.x, s.y, 0)] = p_cat

        if a == A.HOVER:
            sp = S(s.x, s.y, b_after)
            outcomes[sp] = outcomes.get(sp, 0.0) + (p_intended + p_drift)
            return outcomes

        intended_d, drift1_d, drift2_d = _MOVE_DELTAS[a]
        for d, p in ((intended_d, p_intended), (drift1_d, p_drift / 2), (drift2_d, p_drift / 2)):
            nx, ny = s.x + d[0], s.y + d[1]
            # si el viento empuja al dron fuera de la grilla o a una zona
            # restringida, el control de vuelo aborta la maniobra y se
            # mantiene en su celda actual (deriva "suave", no hay colision)
            if not is_free(nx, ny):
                nx, ny = s.x, s.y
            sp = S(nx, ny, b_after)
            outcomes[sp] = outcomes.get(sp, 0.0) + p
        return outcomes

    # ---------- Funcion de recompensa r(s,a,s') (Task 1 seccion 3) ----------
    def reward(self, s: S, a: A, sp: S) -> float:
        r_destino = 100.0 if (sp.x, sp.y) == DEST else 0.0

        delta_b = s.b - sp.b  # positivo = bateria consumida, negativo = se cargo
        r_bateria = -0.1 * delta_b

        if sp.b == 0:
            r_seguridad = -150.0
        elif a != A.CARGAR:
            severity = WEATHER_SEVERITY[self.weather_at(s.x, s.y)]
            r_seguridad = -20.0 * severity
        else:
            r_seguridad = 0.0  # estaciones de carga libres de riesgo

        r_tiempo = TIME_COST[a]
        return r_destino + r_bateria + r_seguridad + r_tiempo


Policy = Callable[[S, List[A]], Dict[A, float]]


def uniform_random_policy(s: S, acts: List[A]) -> Dict[A, float]:
    n = len(acts)
    return {a: 1.0 / n for a in acts}


def fixed_policy(s: S, acts: List[A], charge_threshold: int = 50) -> Dict[A, float]:
    """Politica deterministica fija: si esta en una estacion de carga y la
    bateria esta en o por debajo de `charge_threshold`, carga; en otro caso
    se mueve en la direccion que mas reduce la distancia (Chebyshev) hacia
    el destino, y usa Hover como respaldo si no hay movimientos validos."""
    if A.CARGAR in acts and s.b <= charge_threshold:
        return {A.CARGAR: 1.0}

    move_acts = [a for a in acts if a in MOVE_ACTIONS]
    if not move_acts:
        return {A.HOVER: 1.0}

    def dist_after(a: A) -> float:
        dx, dy = ACTION_ARROWS[a]
        nx, ny = s.x + dx, s.y + dy
        return max(abs(DEST[0] - nx), abs(DEST[1] - ny))

    best = min(move_acts, key=lambda a: (dist_after(a), MOVE_ACTIONS.index(a)))
    return {best: 1.0}


def greedy_policy_from_V(env: GridWorld, V: Dict[S, float]) -> Policy:
    """Deriva una politica deterministica greedy respecto a una funcion de
    valor V ya calculada (Bellman de un paso, seleccionando el argmax)."""

    def policy(s: S, acts: List[A]) -> Dict[A, float]:
        best_a, best_q = None, -math.inf
        for a in acts:
            q = sum(
                p * (env.reward(s, a, sp) + env.gamma * V[sp])
                for sp, p in env.transition(s, a).items()
            )
            if q > best_q:
                best_a, best_q = a, q
        return {best_a: 1.0}

    return policy


def iterative_policy_evaluation(
    env: GridWorld, policy: Policy, theta: float = 1e-4, max_iterations: int = 10_000
) -> Tuple[Dict[S, float], int]:
    """Aplica repetidamente el operador de Bellman de evaluacion de politica
    hasta que el cambio maximo en V entre dos iteraciones sea < theta."""
    V: Dict[S, float] = {s: 0.0 for s in env.states}
    iterations = 0

    while True:
        delta = 0.0
        new_V = dict(V)
        for s in env.states:
            if env.is_terminal(s):
                continue
            acts = env.actions(s)
            if not acts:
                continue
            pi_s = policy(s, acts)
            v = 0.0
            for a, prob_a in pi_s.items():
                if prob_a == 0:
                    continue
                for sp, p_sp in env.transition(s, a).items():
                    v += prob_a * p_sp * (env.reward(s, a, sp) + env.gamma * V[sp])
            new_V[s] = v
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        iterations += 1
        if delta < theta or iterations >= max_iterations:
            break

    return V, iterations


class GridWorldHistory:
    """Mantiene el pequeno historial de simulacion (opcional, para depuracion
    manual del MDP fuera de la evaluacion de politicas)."""

    def __init__(self, env: GridWorld, s0: S):
        self.env = env
        self.s = s0
        self.historial: List[Tuple[S, A, S, float]] = []

    def step(self, a: A) -> Tuple[S, float]:
        outcomes = self.env.transition(self.s, a)
        sp = random.choices(list(outcomes.keys()), weights=list(outcomes.values()))[0]
        r = self.env.reward(self.s, a, sp)
        self.historial.append((self.s, a, sp, r))
        self.s = sp
        return sp, r


# ---------------------------------------------------------------------------
# Reporte: tabla de valores y visualizacion de grilla
# ---------------------------------------------------------------------------

def value_table(env: GridWorld, V: Dict[S, float], battery: int = None):
    """Devuelve un DataFrame con V_pi(s) para el nivel de bateria dado
    (por defecto `env.battery_initial`), indexado por y (fila, de arriba
    hacia abajo) y x (columna)."""
    import pandas as pd

    b = env.battery_initial if battery is None else battery
    data = {}
    for x in range(GRID_SIZE):
        col = []
        for y in range(GRID_SIZE):
            if (x, y) in NO_FLY_ZONES:
                col.append(float("nan"))
            else:
                col.append(V[S(x, y, b)])
        data[x] = col
    df = pd.DataFrame(data)
    df.index.name = "y"
    df.columns.name = "x"
    return df.iloc[::-1]  # y=GRID_SIZE-1 arriba, y=0 abajo


def plot_value_and_policy(env: GridWorld, V: Dict[S, float], battery: int = None,
                           title: str = "", figsize=(6.5, 5.5)):
    """Grafica el grid 5x5 coloreado por V_pi(s), para el nivel de bateria
    dado, con las acciones de la politica greedy (derivada de V via un paso
    de Bellman) superpuestas. Zonas de vuelo restringido en negro, destino
    en magenta, estaciones de carga en amarillo."""
    import matplotlib.pyplot as plt

    b = env.battery_initial if battery is None else battery
    greedy = greedy_policy_from_V(env, V)
    fig, ax = plt.subplots(figsize=figsize)

    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in NO_FLY_ZONES:
                grid[y, x] = V[S(x, y, b)]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        grid, origin="lower", cmap=cmap,
        extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) in NO_FLY_ZONES:
                continue
            s = S(x, y, b)
            ax.text(x, y - 0.30, f"{V[s]:.1f}", ha="center", va="center",
                    fontsize=7, color="white")

            acts = env.actions(s)
            if not acts:
                continue
            a = next(iter(greedy(s, acts)))
            if a in MOVE_ACTIONS:
                dx, dy = ACTION_ARROWS[a]
                scale = 0.32 / SQRT2 if a in DIAGONAL_ACTIONS else 0.32
                ax.annotate(
                    "", xy=(x + dx * scale, y + dy * scale), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.6),
                )
            elif a is A.HOVER:
                ax.text(x, y + 0.25, "H", ha="center", va="center",
                        fontsize=11, color="orange", fontweight="bold")
            elif a is A.CARGAR:
                ax.text(x, y + 0.25, "C", ha="center", va="center",
                        fontsize=11, color="lime", fontweight="bold")

    dx_, dy_ = DEST
    ax.add_patch(plt.Rectangle((dx_ - 0.5, dy_ - 0.5), 1, 1, fill=False, edgecolor="magenta", lw=2))
    for cx, cy in CHARGING_STATIONS:
        ax.add_patch(plt.Rectangle((cx - 0.5, cy - 0.5), 1, 1, fill=False, edgecolor="yellow", lw=2))

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="magenta", lw=2, label="Destino"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="yellow", lw=2, label="Estacion de carga"),
        plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="none", label="Zona de vuelo restringido"),
        plt.Line2D([0], [0], color="red", marker=">", linestyle="-", lw=1.6, markersize=6, label="Accion greedy: mover"),
        plt.Line2D([0], [0], color="none", marker="$H$", markerfacecolor="orange", markeredgecolor="orange",
                    markersize=9, label="Accion greedy: Hover"),
        plt.Line2D([0], [0], color="none", marker="$C$", markerfacecolor="lime", markeredgecolor="lime",
                    markersize=9, label="Accion greedy: Cargar"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.25, 1.0),
              fontsize=8, frameon=True)

    fig.tight_layout()
    return fig


def evaluate_and_report(env: GridWorld, policy: Policy, name: str, theta: float = 1e-4, battery: int = None):
    """Corre la evaluacion iterativa para `policy` y produce los tres
    entregables: tabla de valores, iteraciones hasta convergencia, y
    grafica de grid + flechas de la politica greedy sobre V (para el nivel
    de bateria dado, por defecto `env.battery_initial`)."""
    V, iterations = iterative_policy_evaluation(env, policy, theta=theta)
    b = env.battery_initial if battery is None else battery
    table = value_table(env, V, battery=b)
    fig = plot_value_and_policy(env, V, battery=b, title=f"{name} (theta={theta}, {iterations} iteraciones, b={b}%)")
    return V, iterations, table, fig
