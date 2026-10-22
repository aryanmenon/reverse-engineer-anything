"""Node power-envelope allocator.

Given one (clock, watts, tokens/s) curve per GPU, pick a clock for every GPU
so total watts stay under the envelope and total tokens/s is maximal. Greedy
marginal allocation: start everyone at the lowest clock, then repeatedly give
the next clock step to the GPU that buys the most tokens/s per extra watt.
Idle GPUs buy nothing, so their share flows to busy ones.
"""

from __future__ import annotations

import heapq
from typing import Sequence

Curve = Sequence[tuple[int, float, float]]  # (clock MHz, watts, tokens/s), clock ascending


def efficiency_ceiling(curve: Curve, slo_loss: float) -> int:
    """Index of the most tokens-per-joule clock that loses at most `slo_loss`
    throughput against the fastest clock. Idle curves get index 0."""
    best_t = max(t for _, _, t in curve)
    if best_t <= 0.0:
        return 0
    ok = [i for i, (_, _, t) in enumerate(curve) if t >= (1.0 - slo_loss) * best_t]
    return max(ok, key=lambda i: curve[i][2] / curve[i][1])


def allocate(
    envelope_w: float | None,
    curves: Sequence[Curve],
    ceilings: Sequence[int],
    target_tps: float | None = None,
) -> list[int]:
    """Return a curve index per GPU. `envelope_w=None` means no node cap;
    `target_tps` stops buying clocks once the node can serve that demand."""
    picks = [0] * len(curves)
    total = sum(c[0][1] for c in curves)
    served = sum(c[0][2] for c in curves)
    heap: list[tuple[float, int]] = []

    def push(g: int) -> None:
        i = picks[g]
        if i >= ceilings[g]:
            return
        _, p0, t0 = curves[g][i]
        _, p1, t1 = curves[g][i + 1]
        if t1 <= t0:
            return  # a faster clock that serves nothing more is pure waste (idle GPUs)
        dp = p1 - p0
        gain = float("inf") if dp <= 0 else (t1 - t0) / dp
        heapq.heappush(heap, (-gain, g))

    for g in range(len(curves)):
        push(g)
    while heap and (target_tps is None or served < target_tps):
        _, g = heapq.heappop(heap)
        i = picks[g]
        dp = curves[g][i + 1][1] - curves[g][i][1]
        if envelope_w is not None and total + dp > envelope_w:
            continue  # this GPU's next step does not fit; others' smaller steps still may
        picks[g] = i + 1
        total += dp
        served += curves[g][i + 1][2] - curves[g][i][2]
        push(g)
    return picks
