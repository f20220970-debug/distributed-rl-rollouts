"""All experiments for the README (about 20 minutes on 2 CPU cores). Writes results/*.json and docs/results.png.

1. Scaling: throughput with 1-4 actors, for the plain simulator and a heavier one (+2 ms compute per step)
2. Staleness: max_lag 0 (strictly on-policy, stale data dropped) vs 4
3. Fault tolerance + learning: a full training run where actor 0 is killed at 60 s; the trained policy is then
   evaluated on held-out ships against the dispatching rules
"""
import json

import numpy as np

from drl.system import SysConfig, run
from terminal_rl.env import make
from terminal_rl.heuristics import RULES
from terminal_rl.ppo import policy_fn

def score(policy, n=20):
    v = []
    for s in range(n):
        env = make("base")
        env.reset(seed=900_000 + s)
        done, info, rng = False, {}, np.random.default_rng(s)
        while not done:
            _, _, done, _, info = env.step(policy(env, rng))
        v.append(info["crane_moves_per_hour"])
    return round(float(np.mean(v)), 2)



def main():
    out = {}
    # 1. scaling
    out["scaling"] = []
    for work in [0.0, 2.0]:
        for n in [1, 2, 3, 4]:
            _, m = run(SysConfig(n_actors=n, total_steps=16_384, work_per_step_ms=work), log=lambda s: None)
            row = {"work_per_step_ms": work, "actors": n, "decisions_per_s": m["throughput_decisions_per_s"],
                   "policy_lag_mean": m["policy_lag_mean"], "latency_s_median": m["latency_s_median"]}
            out["scaling"].append(row)
            print(row, flush=True)
    # 2. staleness
    out["staleness"] = []
    for lag in [0, 4]:
        _, m = run(SysConfig(n_actors=3, total_steps=16_384, max_lag=lag), log=lambda s: None)
        row = {"max_lag": lag, "decisions_per_s": m["throughput_decisions_per_s"],
               "dropped_pct": round(100 * m["decisions_dropped_stale"] / (m["decisions_dropped_stale"] + m["decisions_consumed"]), 1),
               "policy_lag_mean": m["policy_lag_mean"]}
        out["staleness"].append(row)
        print(row, flush=True)
    # 3. fault tolerance + learning
    net, m = run(SysConfig(n_actors=2, total_steps=300_000, crash_actor_at_s=60), log=print)
    out["training_run"] = m


    out["policy_quality_base_scenario"] = {"distributed_ppo": score(policy_fn(net)),
                                           **{k: score(RULES[k]) for k in ["fewest_assigned", "most_starved", "dedicated", "random"]}}
    print(out["policy_quality_base_scenario"])
    json.dump(out, open("results/results.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for work, col in [(0.0, "#2e75b6"), (2.0, "#c00000")]:
        r = [x for x in out["scaling"] if x["work_per_step_ms"] == work]
        ax[0].plot([x["actors"] for x in r], [x["decisions_per_s"] for x in r], "o-", color=col,
                   label="plain simulator" if work == 0 else "heavier simulator (+2 ms/step)")
    ax[0].axvline(m["cpu_count"], ls=":", color="k")
    ax[0].text(m["cpu_count"] + 0.05, ax[0].get_ylim()[1] * 0.9, "CPU cores", fontsize=8)
    ax[0].set(title="Throughput vs number of actors", xlabel="actor processes", ylabel="decisions per second")
    ax[0].legend(fontsize=8)
    tl = m["timeline"]
    t = np.array([x["t"] for x in tl]); c = np.array([x["consumed"] for x in tl])
    rate = np.gradient(c, t)
    ax[1].plot(t, rate, color="#2e75b6")
    for r in m["restarts"]:
        ax[1].axvline(r["t"], color="#c00000", ls="--")
    ax[1].axvline(60, color="k", ls=":")
    ax[1].set(title="Fault injection: actor killed at 60 s", xlabel="seconds", ylabel="decisions/s consumed")
    ax[2].plot(t, [x["reward_per_chunk"] for x in tl], color="#548235")
    ax[2].set(title="Learning during the run", xlabel="seconds", ylabel="reward per 256-decision chunk")
    plt.tight_layout()
    plt.savefig("docs/results.png", dpi=130)


if __name__ == "__main__":
    main()
