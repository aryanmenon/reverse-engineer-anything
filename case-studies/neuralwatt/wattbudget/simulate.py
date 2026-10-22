"""Replay Neuralwatt's two headline claims against the simulated GPU model.

    python3 -m wattbudget.simulate

1. Same envelope, more GPUs: 6 stock GPUs vs 8 GPUs inside the same
   6 x 700 W envelope (static per-GPU cap vs the wattbudget agent).
2. Same GPUs, fewer joules: 8 GPUs with no node cap, stock vs agent.

The agent starts from a deliberately wrong prior so the Learn step matters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Mapping

from .gpu_model import DECODE, PHASES, GpuSpec, Phase, clock_under_cap, power, tps
from .optimizer import NodeOptimizer, Telemetry

POLICIES = ("stock", "static-cap", "wattbudget")

# What a pre-trained model might believe before seeing this workload:
# decode a bit less memory-bound and a bit faster than it really is.
BIASED_PRIOR: Mapping[str, Phase] = {
    **PHASES,
    "decode": replace(DECODE, mem_bound=0.55, mem_util=0.80, peak_tps=DECODE.peak_tps * 1.1),
}


@dataclass
class Result:
    policy: str
    n_gpus: int
    envelope_w: float | None
    tokens: float = 0.0
    joules: float = 0.0
    peak_w: float = 0.0
    idle_joules: float = 0.0
    idle_gpu_seconds: float = 0.0
    over_envelope_ticks: int = 0
    backlog_tokens: float = 0.0
    peak_tps: float = 0.0  # best 10 s window, the number Neuralwatt quotes

    @property
    def tokens_per_joule(self) -> float:
        return self.tokens / self.joules

    @property
    def idle_w(self) -> float:
        return self.idle_joules / self.idle_gpu_seconds if self.idle_gpu_seconds else float("nan")


def run(
    policy: str,
    n_gpus: int,
    envelope_w: float | None,
    seconds: float = 600.0,
    dt: float = 0.1,
    mean_demand: float = 24000.0,
    swing: float = 0.35,
    period_s: float = 120.0,
    prior: Mapping[str, Phase] = BIASED_PRIOR,
    seed: int = 1,
) -> Result:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    spec = GpuSpec()
    rng = random.Random(seed)
    phases = ["decode"] * n_gpus
    opt = NodeOptimizer(spec, n_gpus, envelope_w, prior=prior, seed=seed) if policy == "wattbudget" else None
    static_cap = spec.tdp if envelope_w is None else min(spec.tdp, envelope_w / n_gpus)
    res = Result(policy, n_gpus, envelope_w)
    backlog = 0.0
    window: list[float] = []
    window_len = round(10.0 / dt)

    for k in range(round(seconds / dt)):
        t = k * dt
        backlog += mean_demand * (1 + swing * math.sin(2 * math.pi * t / period_s)) * dt
        for g, ph in enumerate(phases):
            if ph == "prefill" and rng.random() < 0.25:
                phases[g] = "decode"
            elif ph == "decode" and rng.random() < 0.03:
                phases[g] = "prefill"

        if opt is None:
            clocks = [spec.f_max] * n_gpus
            limits = [static_cap if policy == "static-cap" else spec.tdp] * n_gpus
        else:
            plan = opt.plan(queued_tokens=backlog, dt=dt)
            clocks, limits = plan.clocks, plan.power_limits
        limits = [min(spec.tdp, max(spec.p_limit_min, w)) for w in limits]

        telemetry: list[Telemetry | None] = [None] * n_gpus
        actual = [0] * n_gpus
        node_w = 0.0
        order = list(range(n_gpus))
        rng.shuffle(order)  # the load balancer does not favour any GPU
        for g in order:
            ph = PHASES[phases[g]]
            f = min(clocks[g], clock_under_cap(spec, limits[g], ph))
            capacity = tps(spec, f, ph) * dt
            served = min(backlog, capacity)
            backlog -= served
            busy = served / capacity if capacity else 0.0
            w = power(spec, f, ph, busy)
            node_w += w
            res.tokens += served
            res.joules += w * dt
            if busy == 0.0:
                res.idle_joules += w * dt
                res.idle_gpu_seconds += dt
            actual[g] = f
            telemetry[g] = Telemetry(
                sm_activity=busy * ph.activity,
                mem_util=busy * ph.mem_util,
                watts=w * (1 + rng.gauss(0, 0.02)),
                tokens_per_s=served / dt,
            )
        res.peak_w = max(res.peak_w, node_w)
        window.append(sum(tel.tokens_per_s for tel in telemetry if tel is not None))
        if len(window) > window_len:
            window.pop(0)
        res.peak_tps = max(res.peak_tps, sum(window) / len(window))
        if envelope_w is not None and node_w > envelope_w + 1e-6:
            res.over_envelope_ticks += 1
        if opt is not None:
            opt.observe(telemetry, actual)  # type: ignore[arg-type]

    res.backlog_tokens = backlog
    return res


def _table(rows: list[Result], baseline: Result) -> str:
    out = [
        "| policy | GPUs | tokens served | vs baseline | peak tokens/s | vs baseline | tokens/J | vs baseline | peak node W | idle W/GPU | ticks over envelope |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.policy} | {r.n_gpus} | {r.tokens / 1e6:.2f} M | {r.tokens / baseline.tokens - 1:+.1%} "
            f"| {r.peak_tps:.0f} | {r.peak_tps / baseline.peak_tps - 1:+.1%} "
            f"| {r.tokens_per_joule:.2f} | {r.tokens_per_joule / baseline.tokens_per_joule - 1:+.1%} "
            f"| {r.peak_w:.0f} | {r.idle_w:.0f} | {r.over_envelope_ticks} |"
        )
    return "\n".join(out)


def main() -> None:
    envelope = 6 * GpuSpec().tdp
    a = run("stock", 6, envelope)
    b = run("static-cap", 8, envelope)
    c = run("wattbudget", 8, envelope)
    print(f"## 1. Same {envelope:.0f} W envelope, demand above 6-GPU capacity at peaks\n")
    print(_table([a, b, c], a))

    d = run("stock", 8, None, mean_demand=19000, swing=0.45)
    e = run("wattbudget", 8, None, mean_demand=19000, swing=0.45)
    print("\n## 2. Same 8 GPUs, no node cap, demand within capacity\n")
    print(_table([d, e], d))
    print(f"\nUnserved backlog at end: stock {d.backlog_tokens:.0f} tokens, wattbudget {e.backlog_tokens:.0f} tokens")


if __name__ == "__main__":
    main()
