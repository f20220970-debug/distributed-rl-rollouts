"""Dispatching rules used as baselines. Each takes the env (read-only) and returns a crane index."""
import numpy as np


def _valid(env):
    return np.flatnonzero(env._valid())


def random_rule(env, rng):
    return int(rng.choice(_valid(env)))


def nearest(env, rng=None):
    truck = env.waiting_trucks[0]
    v = _valid(env)
    return int(v[np.argmin([env.travel_time(truck, c) for c in v])])


def dedicated(env, rng=None):
    """Classic 'gang' practice: each truck is tied to one crane (truck i -> crane i mod C)."""
    truck = env.waiting_trucks[0]
    c = truck % env.cfg.n_cranes
    return c if env._valid()[c] else nearest(env)


def fewest_assigned(env, rng=None):
    truck = env.waiting_trucks[0]
    v = _valid(env)
    key = [(env.c_trucks_assigned[c], env.travel_time(truck, c)) for c in v]
    return int(v[min(range(len(v)), key=key.__getitem__)])


def most_starved(env, rng=None):
    """Send the truck where it prevents the most crane idle time: compare when each crane will next need a
    truck with when its first assigned truck arrives; prefer cranes that will otherwise wait longest."""
    truck, now = env.waiting_trucks[0], env.now
    v = _valid(env)
    best, best_key = v[0], None
    for c in v:
        need_at = now if (env.c_holding[c] and not env.c_queue[c]) else env.c_ready_at[c]
        etas = sorted(e for e in env.c_eta[c] if e > now)
        covered = len(env.c_queue[c]) + len(etas)
        first_cover = now if env.c_queue[c] else (etas[0] if etas else np.inf)
        arrive = now + env.travel_time(truck, c)
        if covered == 0:
            saved = max(0.0, 1800.0 - max(arrive - need_at, 0.0))      # crane has no truck coming at all
        else:
            saved = max(0.0, first_cover - max(need_at, arrive)) - 120.0 * (covered - 1)
        key = (saved, -env.c_trucks_assigned[c], -env.travel_time(truck, c))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return int(best)


RULES = {"random": random_rule, "nearest": nearest, "dedicated": dedicated,
         "fewest_assigned": fewest_assigned, "most_starved": most_starved}


def public_board(env, rng=None):
    """Decentralised greedy rule: a truck sees only the public crane board (idle now / next ready time) and
    its own travel times, not where other trucks are going. It picks the crane it can serve with the least
    crane waiting. Many trucks can make the same choice at once (herding)."""
    truck, now = env.waiting_trucks[0], env.now
    v = _valid(env)
    best, best_key = v[0], None
    for c in v:
        need_at = now if (env.c_holding[c] and not env.c_queue[c]) else env.c_ready_at[c]
        arrive = now + env.travel_time(truck, c)
        key = (-max(arrive - need_at, 0.0) - 0.5 * max(need_at - arrive, 0.0), -env.travel_time(truck, c))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return int(best)


RULES_LOCAL = {"public_board": public_board}
