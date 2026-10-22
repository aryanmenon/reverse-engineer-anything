"""The per-node agent: learn each GPU's power/throughput curve online and
re-plan clocks every tick (100 ms) under the node power envelope.

Mirrors Neuralwatt's published "Connect -> Learn -> Optimize" loop:
  * Learn: CurveModel starts from a prior ("pre-trained efficiency model")
    and corrects it with an EMA of observed watts and tokens/s.
  * Optimize: classify each GPU's workload phase from telemetry, then hand
    the curves to the envelope allocator. Idle GPUs drop to minimum clocks.
  * Backstop: each GPU also gets a hardware power limit just above its
    planned draw, and the node budget is trimmed whenever measured power
    overshoots the envelope.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .budget import allocate, efficiency_ceiling
from .gpu_model import PHASES, GpuSpec, Phase, power, tps


@dataclass(frozen=True)
class Telemetry:
    sm_activity: float  # NVML utilization.gpu / 100
    mem_util: float  # NVML utilization.memory / 100
    watts: float  # NVML power usage
    tokens_per_s: float  # from the inference server's metrics


@dataclass(frozen=True)
class Plan:
    clocks: list[int]
    power_limits: list[float]


def classify(t: Telemetry) -> str:
    """Decode streams weights and KV cache, so it is memory-heavy per unit of SM work."""
    if t.sm_activity < 0.05:
        return "idle"
    return "decode" if t.mem_util / t.sm_activity >= 0.6 else "prefill"


class CurveModel:
    def __init__(self, spec: GpuSpec, prior: Mapping[str, Phase] = PHASES, alpha: float = 0.2):
        self.levels = spec.levels()
        self.alpha = alpha
        self._est = {
            (name, f): [power(spec, f, phase), tps(spec, f, phase)]
            for name, phase in prior.items()
            for f in self.levels
        }

    def curve(self, state: str) -> list[tuple[int, float, float]]:
        return [(f, *self._est[(state, f)]) for f in self.levels]

    def observe(self, state: str, f: int, watts: float, tokens_per_s: float) -> None:
        f = min(self.levels, key=lambda level: abs(level - f))  # real SM clocks land between levels
        est = self._est[(state, f)]
        est[0] += self.alpha * (watts - est[0])
        est[1] += self.alpha * (tokens_per_s - est[1])


class NodeOptimizer:
    def __init__(
        self,
        spec: GpuSpec,
        n_gpus: int,
        envelope_w: float | None = None,
        slo_loss: float = 0.05,
        explore: float = 0.02,
        limit_margin: float = 0.03,
        demand_headroom: float = 1.05,
        prior: Mapping[str, Phase] = PHASES,
        seed: int = 0,
    ):
        self.spec = spec
        self.envelope_w = envelope_w
        self.slo_loss = slo_loss
        self.explore = explore
        self.limit_margin = limit_margin
        self.demand_headroom = demand_headroom
        self.models = [CurveModel(spec, prior) for _ in range(n_gpus)]
        self.states = ["idle"] * n_gpus
        self._budget = envelope_w
        self._rng = random.Random(seed)

    def plan(self, queued_tokens: float, dt: float) -> Plan:
        """Clocks and power limits for the next tick. `queued_tokens` comes from
        the inference server's queue; the agent buys just enough clock to drain
        it, so spare GPUs sit at minimum clocks instead of idling at boost."""
        # Work is about to land on idle GPUs too: plan them as the common phase.
        states = ["decode" if s == "idle" and queued_tokens > 0 else s for s in self.states]
        curves = [m.curve(s) for m, s in zip(self.models, states)]
        target = self.demand_headroom * queued_tokens / dt
        ceilings = [efficiency_ceiling(c, self.slo_loss) for c in curves]
        if sum(c[i][2] for c, i in zip(curves, ceilings)) < target:
            # The queue is outgrowing efficient clocks: throughput comes first.
            ceilings = [len(c) - 1 for c in curves]
        budget = None if self._budget is None else self._budget / (1 + self.limit_margin)
        picks = allocate(budget, curves, ceilings, target_tps=target)
        for g, s in enumerate(states):
            if s != "idle" and picks[g] > 0 and self._rng.random() < self.explore:
                picks[g] -= 1  # explore downward only, so the envelope stays safe
        clocks = [curves[g][i][0] for g, i in enumerate(picks)]
        # Hardware limit just above the planned draw: if a GPU switches to a
        # hungrier phase mid-tick, its own limiter downclocks it instead of
        # the node blowing through the envelope.
        limits = [curves[g][i][1] * (1 + self.limit_margin) for g, i in enumerate(picks)]
        return Plan(clocks, limits)

    def observe(self, telemetry: Sequence[Telemetry], clocks: Sequence[int]) -> None:
        for g, (t, f) in enumerate(zip(telemetry, clocks)):
            state = classify(t)
            self.states[g] = state
            if state != "idle" and t.sm_activity >= 0.95:
                self.models[g].observe(state, f, t.watts, t.tokens_per_s)
        if self.envelope_w is not None:
            measured = sum(t.watts for t in telemetry)
            # Integral trim: shrink the planning budget after an overshoot, relax back slowly.
            self._budget = min(self.envelope_w, max(0.9 * self.envelope_w, self._budget + 0.5 * (self.envelope_w - measured)))
