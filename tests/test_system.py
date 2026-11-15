"""Short end-to-end runs of the actor-learner system (about 30 s)."""
from drl.system import SysConfig, run


def test_consumes_requested_steps_and_publishes_versions():
    _, m = run(SysConfig(n_actors=2, total_steps=4096), log=lambda s: None)
    assert m["decisions_consumed"] >= 4096
    assert m["updates"] >= 2
    assert m["policy_lag_mean"] <= 4


def test_supervisor_restarts_a_killed_actor():
    _, m = run(SysConfig(n_actors=2, total_steps=8192, crash_actor_at_s=3), log=lambda s: None)
    assert any(r["actor"] == 0 for r in m["restarts"])
    assert m["decisions_consumed"] >= 8192
