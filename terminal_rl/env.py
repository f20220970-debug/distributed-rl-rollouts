"""Container terminal horizontal-transport simulator (Gymnasium environment).

A vessel is being unloaded by quay cranes. Each crane lifts a container off the ship and must hand it to an
internal truck; if no truck is waiting under it, the crane sits idle. Trucks carry each container to a yard
block, queue for the yard crane there, unload, and then need a new job. The decision is made every time a
truck becomes empty: WHICH QUAY CRANE SHOULD IT DRIVE TO NEXT?

Event-driven (semi-Markov): one environment step = one dispatch decision; simulated time jumps between them.

Reward = minus the crane idle time (in minutes) that accrued since the previous decision, so the episode return
is minus total crane idle time. Less idle time = higher crane productivity = shorter vessel turnaround.

Everything is synthetic. Cycle times and distances are in plausible ranges for a mid-size terminal, but they
are assumptions (see Config), not data from any real terminal.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Config:
    n_cranes: int = 4
    n_trucks: int = 10
    n_blocks: int = 6
    containers_per_crane: int = 60
    crane_cycle_mean_s: float = 110.0      # quay crane lift cycle (ship -> handover point)
    crane_cycle_cv: float = 0.30
    handover_s: float = 15.0
    yard_service_s: tuple = (50.0, 140.0)  # yard crane unload time, uniform
    speed_m_s: float = 6.0                 # truck speed in the terminal
    crane_spacing_m: float = 120.0
    block_offset_m: float = 300.0
    crane_block_skew: float = 0.6          # share of a crane's boxes going to its two "home" blocks
    breakdown: bool = False                # stress test: one crane stops for a while mid-shift
    breakdown_s: float = 1500.0
    max_steps: int = 5000
    feats_per_crane: int = 7
    makespan_coef: float = 0.0             # optional end-of-episode penalty on vessel turnaround (ablation)
    obs_mode: str = "full"                 # "full" = central dispatcher sees every truck's plan
                                           # "local" = each truck-agent sees only the public crane board
                                           # "comm" = local + trucks broadcast their chosen crane (counts only)


class TerminalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_cranes
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(C * cfg.feats_per_crane + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(C)
        # geometry: cranes along the quay (x), blocks behind the quay at y = block_offset
        self.crane_x = np.arange(C) * cfg.crane_spacing_m
        span = self.crane_x[-1] if C > 1 else cfg.crane_spacing_m
        self.block_x = np.linspace(-0.1 * span, 1.1 * span, cfg.n_blocks)
        self.d_cb = np.hypot(self.crane_x[:, None] - self.block_x[None, :], cfg.block_offset_m)  # metres
        self.t_cb = self.d_cb / cfg.speed_m_s
        self.t_depot = np.hypot(self.crane_x - span / 2, cfg.block_offset_m * 0.5) / cfg.speed_m_s
        # destination block distribution per crane (skewed towards the two nearest blocks)
        P = np.full((C, cfg.n_blocks), (1 - cfg.crane_block_skew) / cfg.n_blocks)
        for c in range(C):
            home = np.argsort(self.d_cb[c])[:2]
            P[c, home] += cfg.crane_block_skew / 2
        self.dest_p = P / P.sum(1, keepdims=True)

    # ------------------------------------------------------------------ simulation core
    def _cycle(self):
        m, cv = self.cfg.crane_cycle_mean_s, self.cfg.crane_cycle_cv
        s2 = np.log(1 + cv ** 2)
        return float(self.rng.lognormal(np.log(m) - s2 / 2, np.sqrt(s2)))

    def _push(self, t, kind, *args):
        heapq.heappush(self.events, (t, self._seq, kind, args))
        self._seq += 1

    def _accrue_idle(self, t):
        for c in range(self.cfg.n_cranes):
            if self.c_holding[c] and not self.c_queue[c]:
                self.idle_total += t - max(self.c_idle_from[c], self.last_t)
                self.c_idle_from[c] = t
        self.last_t = t

    def _start_lift(self, c, t):
        if self.c_remaining[c] <= 0:
            self.c_done_at[c] = t
            return
        self.c_remaining[c] -= 1
        dur = self._cycle()
        if self.cfg.breakdown and c == self.bd_crane and self.bd_start <= t < self.bd_start + self.cfg.breakdown_s:
            dur += self.bd_start + self.cfg.breakdown_s - t
        self.c_ready_at[c] = t + dur
        self._push(t + dur, "lift_done", c)

    def _handover(self, c, t):
        truck = self.c_queue[c].pop(0)
        self.c_holding[c] = False
        self.c_trucks_assigned[c] -= 1
        b = int(self.rng.choice(self.cfg.n_blocks, p=self.dest_p[c]))
        self._push(t + self.cfg.handover_s + self.t_cb[c, b], "truck_at_yard", truck, b)
        self._start_lift(c, t + self.cfg.handover_s)

    def _process(self, t, kind, args):
        self._accrue_idle(t)
        self.now = t
        if kind == "lift_done":
            (c,) = args
            self.c_holding[c] = True
            self.c_idle_from[c] = t
            if self.c_queue[c]:
                self._handover(c, t)
        elif kind == "truck_at_crane":
            truck, c = args
            self.c_queue[c].append(truck)
            if self.c_holding[c] and len(self.c_queue[c]) == 1:
                self._handover(c, t)
        elif kind == "truck_at_yard":
            truck, b = args
            start = max(t, self.b_free_at[b])
            done = start + self.rng.uniform(*self.cfg.yard_service_s)
            self.b_free_at[b] = done
            self._push(done, "truck_empty", truck, b)
        elif kind == "truck_empty":
            truck, b = args
            self.delivered += 1
            self.t_loc[truck] = b
            self.waiting_trucks.append(truck)

    def _valid(self):
        return (self.c_remaining + self.c_holding.astype(int)) > 0

    def _advance(self):
        """Run events until a truck needs a decision (and some crane still needs trucks) or the vessel is done."""
        while True:
            if self.waiting_trucks and self._valid().any():
                return False
            if self.delivered >= self.total:
                return True
            if not self.events:
                return True
            t, _, kind, args = heapq.heappop(self.events)
            self._process(t, kind, args)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        cfg, C = self.cfg, self.cfg.n_cranes
        self.events, self._seq = [], 0
        self.now = self.last_t = 0.0
        self.idle_total = 0.0
        self.total = C * cfg.containers_per_crane
        self.delivered = 0
        self.c_remaining = np.full(C, cfg.containers_per_crane, dtype=int)
        self.c_holding = np.zeros(C, dtype=bool)
        self.c_queue = [[] for _ in range(C)]
        self.c_idle_from = np.zeros(C)
        self.c_ready_at = np.zeros(C)
        self.c_done_at = np.full(C, np.nan)
        self.c_trucks_assigned = np.zeros(C, dtype=int)
        self.c_eta = [[] for _ in range(C)]          # arrival times of trucks driving to each crane
        self.b_free_at = np.zeros(cfg.n_blocks)
        self.t_loc = np.full(cfg.n_trucks, -1)       # -1 = depot
        self.waiting_trucks = list(range(cfg.n_trucks))
        self.bd_crane = int(self.rng.integers(C))
        self.bd_start = float(self.rng.uniform(600, 2400))
        self.steps = 0
        self._idle_at_last_decision = 0.0
        for c in range(C):
            self._start_lift(c, 0.0)
        self._advance()
        return self._obs(), {"action_mask": self._valid()}

    def travel_time(self, truck, c):
        loc = self.t_loc[truck]
        return self.t_depot[c] if loc < 0 else self.t_cb[c, loc]

    def _obs(self):
        cfg, C = self.cfg, self.cfg.n_cranes
        truck = self.waiting_trucks[0] if self.waiting_trucks else 0
        f = np.zeros((C, cfg.feats_per_crane), dtype=np.float32)
        for c in range(C):
            etas = [e for e in self.c_eta[c] if e > self.now]
            next_eta = min(etas) if etas else self.now + 900
            ready_in = max(self.c_ready_at[c] - self.now, 0.0)
            arrive_in = self.travel_time(truck, c)
            f[c] = [
                (self.c_remaining[c] + self.c_holding[c]) / cfg.containers_per_crane,
                float(self.c_holding[c] and not self.c_queue[c]),          # crane idle right now
                ready_in / 300.0,
                self.c_trucks_assigned[c] / 4.0,
                (next_eta - self.now - ready_in) / 300.0,                   # slack: +ve = crane will wait
                arrive_in / 300.0,
                (arrive_in - ready_in) / 300.0,                             # would THIS truck be early or late
            ]
        if cfg.obs_mode in ("local", "comm"):
            f[:, 4] = 0.0                      # no knowledge of other trucks' arrival times
            if cfg.obs_mode == "local":
                f[:, 3] = 0.0                  # no knowledge of how many trucks are heading to each crane
        g = np.array([len(self.waiting_trucks) / cfg.n_trucks, self.delivered / self.total,
                      self.steps / 1000.0], dtype=np.float32)
        return np.clip(np.concatenate([f.ravel(), g]), -5, 5)

    def step(self, action):
        action = int(action)
        valid = self._valid()
        if not valid[action]:                       # safety: never happens with masking
            action = int(np.flatnonzero(valid)[0])
        truck = self.waiting_trucks.pop(0)
        arr = self.now + self.travel_time(truck, action)
        self.c_trucks_assigned[action] += 1
        self.c_eta[action] = [e for e in self.c_eta[action] if e > self.now] + [arr]
        self._push(arr, "truck_at_crane", truck, action)
        self.steps += 1
        done = self._advance() or self.steps >= self.cfg.max_steps
        if done:  # flush remaining events so idle time and makespan are complete
            while self.events:
                t, _, kind, args = heapq.heappop(self.events)
                self._process(t, kind, args)
        reward = -(self.idle_total - self._idle_at_last_decision) / 60.0
        self._idle_at_last_decision = self.idle_total
        if done and self.cfg.makespan_coef > 0:
            # minutes of turnaround above a simple lower bound (every crane busy all the time)
            lb = self.cfg.containers_per_crane * (self.cfg.crane_cycle_mean_s + self.cfg.handover_s) / 60.0
            reward -= self.cfg.makespan_coef * max(0.0, self.now / 60.0 - lb)
        info = {"action_mask": self._valid() if not done else np.ones(self.cfg.n_cranes, bool)}
        if done:
            info.update(self.kpis())
        return self._obs(), reward, done, False, info

    def kpis(self):
        C = self.cfg.n_cranes
        makespan = self.now
        moves = self.total - int(self.c_remaining.sum())
        return {
            "makespan_h": makespan / 3600,
            "crane_moves_per_hour": moves / (C * makespan / 3600),
            "crane_idle_pct": 100 * self.idle_total / (C * makespan),
            "delivered": self.delivered,
        }


def make(scenario: str = "base") -> TerminalEnv:
    """Named scenarios used in training and the held-out evaluation."""
    base = Config()
    cfgs = {
        "base": base,
        "fewer_trucks": replace(base, n_trucks=8),
        "more_trucks": replace(base, n_trucks=14),
        "breakdown": replace(base, breakdown=True),
        "volatile_cranes": replace(base, crane_cycle_cv=0.6),
        "six_cranes": replace(base, n_cranes=6, n_trucks=15),
    }
    return TerminalEnv(cfgs[scenario])
