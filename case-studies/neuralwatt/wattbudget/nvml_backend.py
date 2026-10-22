"""Real-hardware backend: drive NodeOptimizer from NVML on an NVIDIA node.

Not exercised by the tests (no GPU in CI). Needs `pip install nvidia-ml-py`,
root (or admin clock permissions), and a tokens/s source from your inference
server, e.g. the delta of vLLM's `vllm:generation_tokens_total` per GPU.

    NVML calls this maps to:
      read   nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates,
             nvmlDeviceGetClockInfo(SM), nvmlDeviceGetTotalEnergyConsumption
      write  nvmlDeviceSetGpuLockedClocks, nvmlDeviceSetPowerManagementLimit
      undo   nvmlDeviceResetGpuLockedClocks, default power limit
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

from .optimizer import NodeOptimizer, Telemetry


class NvmlNode:
    def __init__(self) -> None:
        import pynvml  # provided by nvidia-ml-py

        self.nv = pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self.limit_range = [pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h) for h in self.handles]
        self.default_limits = [pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) for h in self.handles]

    def telemetry(self, tokens_per_s: Sequence[float]) -> list[Telemetry]:
        out = []
        for h, tps in zip(self.handles, tokens_per_s):
            util = self.nv.nvmlDeviceGetUtilizationRates(h)
            out.append(
                Telemetry(
                    sm_activity=util.gpu / 100,
                    mem_util=util.memory / 100,
                    watts=self.nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    tokens_per_s=tps,
                )
            )
        return out

    def sm_clocks(self) -> list[int]:
        return [self.nv.nvmlDeviceGetClockInfo(h, self.nv.NVML_CLOCK_SM) for h in self.handles]

    def energy_joules(self) -> float:
        return sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles) / 1000

    def apply(self, f_min: int, clocks: Sequence[int], limits_w: Sequence[float]) -> None:
        for h, (lo, hi), f, w in zip(self.handles, self.limit_range, clocks, limits_w):
            self.nv.nvmlDeviceSetGpuLockedClocks(h, f_min, f)
            self.nv.nvmlDeviceSetPowerManagementLimit(h, int(min(hi, max(lo, w * 1000))))

    def reset(self) -> None:
        for h, w in zip(self.handles, self.default_limits):
            self.nv.nvmlDeviceResetGpuLockedClocks(h)
            self.nv.nvmlDeviceSetPowerManagementLimit(h, w)


def run_agent(
    node: NvmlNode,
    opt: NodeOptimizer,
    tokens_per_s: Callable[[], Sequence[float]],
    queued_tokens: Callable[[], float],
    period_s: float = 0.1,
) -> None:
    """100 ms control loop; restores stock clocks and limits on exit."""
    try:
        while True:
            start = time.monotonic()
            plan = opt.plan(queued_tokens(), period_s)
            node.apply(opt.spec.f_min, plan.clocks, plan.power_limits)
            time.sleep(max(0.0, period_s - (time.monotonic() - start)))
            opt.observe(node.telemetry(tokens_per_s()), node.sm_clocks())
    finally:
        node.reset()
