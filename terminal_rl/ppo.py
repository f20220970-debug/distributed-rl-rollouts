"""PPO written from scratch in PyTorch (no RL library), with action masking.

Policy design: the same small network scores every crane from that crane's features plus a few global ones,
so the policy does not depend on how many cranes there are (it can be run on a 6-crane quay after training
on 4). The critic pools the per-crane embeddings (mean and max) to estimate the value of the state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from .env import Config, TerminalEnv


class CraneScorer(nn.Module):
    def __init__(self, f_crane: int, f_global: int, h: int = 64):
        super().__init__()
        self.fc, self.fg = f_crane, f_global
        self.embed = nn.Sequential(nn.Linear(f_crane + f_global, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, 1)
        self.v = nn.Sequential(nn.Linear(2 * h + f_global, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, obs: torch.Tensor, n_cranes: int, mask: torch.Tensor):
        B = obs.shape[0]
        crane = obs[:, : n_cranes * self.fc].view(B, n_cranes, self.fc)
        glob = obs[:, n_cranes * self.fc:]
        e = self.embed(torch.cat([crane, glob[:, None, :].expand(B, n_cranes, self.fg)], -1))
        logits = self.pi(e).squeeze(-1).masked_fill(~mask, -1e9)
        value = self.v(torch.cat([e.mean(1), e.max(1).values, glob], -1)).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    total_steps: int = 250_000
    rollout: int = 4096
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad: float = 0.5
    seed: int = 0
    makespan_coef: float = 0.0     # 0 = idle-time reward only (default); >0 adds the turnaround penalty
    obs_mode: str = "full"         # "full" (central dispatcher) | "local" | "comm" (decentralised truck agents)


def train_env_sampler(rng, makespan_coef=0.0, obs_mode="full"):
    """Domain randomisation during training: truck fleet 8-14, crane cycle variability 0.25-0.5 (4 cranes).
    Breakdowns and 6-crane quays are never seen in training (they are held-out stress tests)."""
    return replace(Config(), n_trucks=int(rng.integers(8, 15)), crane_cycle_cv=float(rng.uniform(0.25, 0.5)),
                   makespan_coef=makespan_coef, obs_mode=obs_mode)


def train(cfg: PPOConfig = PPOConfig(), log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = TerminalEnv(train_env_sampler(rng, cfg.makespan_coef, cfg.obs_mode))
    C, F = env.cfg.n_cranes, env.cfg.feats_per_crane
    net = CraneScorer(F, 3)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    obs, info = env.reset(seed=int(rng.integers(1e9)))
    mask = info["action_mask"]
    history, ep_ret, ep_rets, t0, step = [], 0.0, [], time.time(), 0
    while step < cfg.total_steps:
        O, M, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(cfg.rollout):
            o = torch.as_tensor(obs, dtype=torch.float32)[None]
            m = torch.as_tensor(mask)[None]
            with torch.no_grad():
                logits, v = net(o, C, m)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs2, r, done, _, info = env.step(a.item())
            O.append(o[0]); M.append(m[0]); A.append(a[0]); LP.append(dist.log_prob(a)[0]); V.append(v[0])
            R.append(r); D.append(done)
            ep_ret += r
            step += 1
            if done:
                ep_rets.append(ep_ret); ep_ret = 0.0
                env.cfg = train_env_sampler(rng, cfg.makespan_coef, cfg.obs_mode)   # same quay (4 cranes), new fleet / volatility
                obs, info = env.reset(seed=int(rng.integers(1e9)))
            else:
                obs = obs2
            mask = info["action_mask"]
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs, dtype=torch.float32)[None], C, torch.as_tensor(mask)[None])
        # GAE
        R_t, D_t, V_t = torch.tensor(R), torch.tensor(D, dtype=torch.float32), torch.stack(V)
        adv, gae = torch.zeros_like(R_t), 0.0
        for t in reversed(range(len(R))):
            nv = last_v[0] if t == len(R) - 1 else V_t[t + 1]
            delta = R_t[t] + cfg.gamma * nv * (1 - D_t[t]) - V_t[t]
            gae = delta + cfg.gamma * cfg.lam * (1 - D_t[t]) * gae
            adv[t] = gae
        ret = adv + V_t
        O_t, M_t, A_t, LP_t = torch.stack(O), torch.stack(M), torch.stack(A), torch.stack(LP)
        for _ in range(cfg.epochs):
            idx = torch.randperm(len(R))
            for s in range(0, len(R), cfg.minibatch):
                b = idx[s:s + cfg.minibatch]
                logits, v = net(O_t[b], C, M_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A_t[b]) - LP_t[b])
                ab = (adv[b] - adv[b].mean()) / (adv[b].std() + 1e-8)
                pg = -torch.min(ratio * ab, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ab).mean()
                vl = ((v - ret[b]) ** 2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
                opt.step()
        recent = float(np.mean(ep_rets[-20:])) if ep_rets else float("nan")
        history.append({"step": step, "mean_return_last20": recent, "minutes": (time.time() - t0) / 60})
        log(f"step {step:>7}  mean episode return (last 20) {recent:8.1f}  ({(time.time() - t0) / 60:.1f} min)")
    return net, history


def policy_fn(net: CraneScorer):
    """Greedy (argmax) policy for evaluation."""
    def act(env: TerminalEnv, rng=None):
        o = torch.as_tensor(env._obs(), dtype=torch.float32)[None]
        m = torch.as_tensor(env._valid())[None]
        with torch.no_grad():
            logits, _ = net(o, env.cfg.n_cranes, m)
        return int(logits.argmax(-1).item())
    return act
