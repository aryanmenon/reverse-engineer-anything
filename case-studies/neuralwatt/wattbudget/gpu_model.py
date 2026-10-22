"""Simulated GPU power/performance model used when no real GPU is present.

The model is plain DVFS physics, calibrated to public H100 SXM5 figures:
~700 W at full boost under load, ~125 W idle at stock clocks, ~65-73 W idle
at minimum clocks (Neuralwatt's Crusoe result: 125 W -> 73 W idle).

    dynamic power  ~ f * V(f)^2   (V rises roughly linearly with f)
    HBM power      ~ memory bandwidth, independent of SM clock
    step time      = compute part (shrinks with f) + memory part (does not)

That last line is the whole trick: LLM decode is memory-bound, so dropping
the SM clock costs little throughput but a lot of power.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    name: str
    activity: float  # SM busy fraction while serving
    mem_util: float  # HBM bandwidth utilisation while serving
    mem_bound: float  # fraction of step time that does not shrink with SM clock
    peak_tps: float  # tokens/s at f_max


IDLE = Phase("idle", 0.0, 0.0, 0.0, 0.0)
PREFILL = Phase("prefill", 1.0, 0.35, 0.10, 12000.0)
DECODE = Phase("decode", 1.0, 0.90, 0.65, 2600.0)
PHASES = {p.name: p for p in (IDLE, PREFILL, DECODE)}


@dataclass(frozen=True)
class GpuSpec:
    f_min: int = 345
    f_max: int = 1980
    f_step: int = 45
    v_min: float = 0.70
    v_max: float = 1.10
    p_static: float = 62.0
    p_idle_dyn: float = 63.0
    p_mem: float = 110.0
    p_core: float = 475.0
    tdp: float = 700.0
    p_limit_min: float = 200.0  # lowest power limit the board accepts

    def levels(self) -> tuple[int, ...]:
        levels = list(range(self.f_min, self.f_max + 1, self.f_step))
        if levels[-1] != self.f_max:
            levels.append(self.f_max)
        return tuple(levels)


def _scale(spec: GpuSpec, f: float) -> float:
    v = spec.v_min + (spec.v_max - spec.v_min) * (f - spec.f_min) / (spec.f_max - spec.f_min)
    return (f / spec.f_max) * (v / spec.v_max) ** 2


def power(spec: GpuSpec, f: float, phase: Phase, busy: float = 1.0) -> float:
    """Board power in watts at clock f, serving `phase` for `busy` of the tick."""
    s = _scale(spec, f)
    load = busy * phase.activity * (spec.p_mem * phase.mem_util + spec.p_core * s)
    return spec.p_static + spec.p_idle_dyn * s + load


def tps(spec: GpuSpec, f: float, phase: Phase) -> float:
    """Tokens/s at clock f while fully busy in `phase`."""
    if phase.peak_tps == 0.0:
        return 0.0
    x = f / spec.f_max
    return phase.peak_tps / ((1.0 - phase.mem_bound) / x + phase.mem_bound)


def clock_under_cap(spec: GpuSpec, cap_w: float, phase: Phase, busy: float = 1.0) -> int:
    """What the driver's power limiter does: highest clock whose power fits the cap."""
    for f in reversed(spec.levels()):
        if power(spec, f, phase, busy) <= cap_w:
            return f
    return spec.f_min
