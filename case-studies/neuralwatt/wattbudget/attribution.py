"""Per-request energy attribution, reconstructed from Neuralwatt's `: energy`
SSE payload and its published methodology.

The captured payload is self-consistent with:

    server_joules  = avg_power_watts (whole 8-GPU server) * duration_seconds
    ratio          = min(request tokens / in-flight token pool, cap)
    energy_joules  = server_joules * ratio

4087 W * 1.479 s * 0.0034 = 20.55 J vs the reported 20.44 J; the payload
rounds the ratio, and 20.44 / 6044.7 = 0.00338.
"""

from __future__ import annotations

J_PER_KWH = 3.6e6


def attribution_ratio(request_tokens: float, pool_tokens: float, cap: float) -> float:
    """Token share of the in-flight pool, capped per hardware class so a lone
    request on a quiet server is not billed for the whole box."""
    if pool_tokens <= 0:
        return cap
    return min(request_tokens / pool_tokens, cap)


def energy_payload(
    server_joules: float,
    duration_s: float,
    request_tokens: float,
    pool_tokens: float,
    n_gpus: int,
    cap: float,
    grid_gco2_per_kwh: float,
    grid_id: str,
) -> dict[str, object]:
    """Build the same fields Neuralwatt streams as its `: energy` SSE comment.
    `server_joules` should be a delta of NVML's cumulative energy counters
    (nvmlDeviceGetTotalEnergyConsumption) summed over the server's GPUs."""
    ratio = attribution_ratio(request_tokens, pool_tokens, cap)
    joules = server_joules * ratio
    kwh = joules / J_PER_KWH
    return {
        "energy_joules": joules,
        "energy_kwh": kwh,
        "avg_power_watts": server_joules / duration_s,
        "duration_seconds": duration_s,
        "attribution_method": f"prorated_token_pool_weighted_multi_gpu_{n_gpus}",
        "attribution_ratio": ratio,
        "carbon_g_co2eq": kwh * grid_gco2_per_kwh,
        "grid_carbon_intensity_gco2perkwhr": grid_gco2_per_kwh,
        "grid_id": grid_id,
    }
