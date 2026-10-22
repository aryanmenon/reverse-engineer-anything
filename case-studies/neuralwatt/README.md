# Neuralwatt teardown: more AI compute under the same power budget

Neuralwatt sells GPU power-management software ("Optimize") plus an
energy-priced inference cloud. Its headline claims:

- **8 GPUs in the power envelope of 6.** Peak token throughput up 33% on a
  Crusoe Cloud DGX H100, with lower peak power.
- **Idle GPU power from 125 W to 73 W.**
- **+17% peak tokens/W.** An illustrative trial: 1.27 → 1.49 tokens/W, peak
  throughput 5,527 tok/s (+23%), peak power 3,862 W (−14%).
- **Under 0.1% performance overhead**, with no hardware changes.

This directory reconstructs how that works and ships a working
reimplementation (`wattbudget/`). It includes a GPU physics simulator, so the
claims can be replayed without hardware.

## What we could inspect

Neuralwatt's controller is closed and server-side, so no binary was
available. Evidence came from three places:

| Source | What it gave | Grade |
|---|---|---|
| Public site, docs, Crusoe case study (via search snippets; direct fetches were blocked in this environment) | Architecture, claims, NVML usage | Vendor statements |
| [`pi-neuralwatt-provider@1.19.0`](https://www.npmjs.com/package/pi-neuralwatt-provider) (MIT, third-party client), `docs/sse-payloads.md` | Real `: energy` / `: cost` SSE payloads and observed flex-tier behaviour | **Observed** wire data |
| Arithmetic on those payloads | Attribution formula | **Derived** |

Labels below follow REA's convention: **observed**, **derived**,
**inferred** (best explanation, not proven) and **unknown**.

## How it works (reconstruction)

### 1. Per-server agent with a ~100 ms control loop (observed + inferred)

The docs describe three runtime parts: an **agent on every GPU server**, an
**orchestrator**, and APIs. The agent collects telemetry, sends it over
WebSocket/REST heartbeats, and performs "local optimization actions (e.g.
CPU frequency, GPU power)". The site says it "adjusts GPU power to the
workload every 100ms". Telemetry comes from **NVML**, sampled several times
per second, and lands in ClickHouse.

**Inferred:** NVML exposes exactly two such actions: locked SM clocks
(`nvmlDeviceSetGpuLockedClocks`) and power limits
(`nvmlDeviceSetPowerManagementLimit`). The 100 ms cadence matches NVML's
power-sample granularity.

### 2. Why lower clocks cost little throughput (derived from physics)

Dynamic power scales with `f · V(f)²`, and voltage rises with frequency, so
the top of the clock range is very expensive. LLM **decode** is
memory-bandwidth-bound: most of each step waits on HBM, and that wait does
not shrink with SM clock. In our calibrated model, dropping an H100 from
1980 MHz to ~1700 MHz keeps 95% of decode throughput and cuts board power
about 18%. **Prefill** is compute-bound and loses throughput almost linearly,
so a good controller has to be **phase-aware**. Neuralwatt's "pre-trained
efficiency models" are most likely these per-phase curves.

### 3. Where "8 in the envelope of 6" comes from (derived)

A 6 × 700 W budget spread over 8 GPUs gives each one 525 W, roughly 81% of
max clock. Each GPU then keeps about 92% of its decode throughput, because
decode does not care much about clock. That is about 7.4 GPUs' worth of
throughput from 6 GPUs' worth of power. On top of that:

- **Idle GPUs give their budget away.** At stock clocks an idle H100 burns
  ~125 W. Locked to minimum clocks it burns ~70 W, and the agent hands that
  freed budget to busy GPUs.
- **The envelope is shared dynamically, not split evenly.** Every tick, the
  next increment of power goes to whichever GPU buys the most tokens per
  extra watt.

### 4. Efficiency mode (observed claim, inferred mechanism)

When the envelope isn't binding, the agent caps each GPU at its most
tokens-per-joule clock that stays within an SLO throughput loss. It also buys
only enough clock to drain the current queue. "Optional Q-learning" in the
docs suggests a learned policy on top of that. **Unknown:** the state and
reward design.

### 5. Per-request energy metering (observed + derived)

Every streamed response ends with an SSE comment like this one
(`evidence/sse-energy-payload.json`):

```json
{"energy_joules": 20.44, "avg_power_watts": 4087.0, "duration_seconds": 1.479,
 "attribution_method": "prorated_token_pool_weighted_multi_gpu_8",
 "attribution_ratio": 0.0034, "grid_id": "FI", "carbon_g_co2eq": 0.0002726}
```

**Derived:** `avg_power_watts` is the **whole 8-GPU server**, not this
request. The numbers reconcile as:

```
server_joules = 4087 W × 1.479 s           = 6044.7 J   (NVML energy-counter delta)
ratio         = min(req tokens / in-flight token pool, cap)   ≈ 0.00338 (rounded to 0.0034)
energy_joules = server_joules × ratio      = 20.44 J
kwh           = 20.44 / 3.6e6              = 5.68e-6
carbon        = kwh × 48 gCO2/kWh (FI)     = 2.73e-4 g
```

The docs mention a per-hardware-class cap, "typically 25%", so a lone
request on a quiet server isn't billed for the whole box.
`tests/test_wattbudget.py` reproduces the captured payload to within 0.2%.

### 6. Flex tier: filling the valleys (observed)

`*-flex` models are held server-side, sending heartbeats every ~10 s; queue
waits of 0 s to ~6 min were observed. They run when capacity frees up and
are billed at 0.65× energy. **Inferred:** this is load-shifting. Deferrable
work fills the idle gaps that would otherwise burn ~70–125 W per GPU for
nothing, which raises tokens per joule at fleet level. (Not implemented
here.)

## The reimplementation: `wattbudget/`

| Module | Role |
|---|---|
| `gpu_model.py` | DVFS power/throughput model calibrated to H100: 700 W TDP, 125 W stock idle, ~70 W idle at minimum clock |
| `budget.py` | Greedy marginal tokens-per-watt allocator under a node envelope, plus an SLO-bounded efficiency ceiling |
| `optimizer.py` | Agent: phase classification from telemetry, online curve learning from a prior, demand-aware planning, power-limit backstop, integral trim on overshoot |
| `attribution.py` | Neuralwatt-compatible per-request `: energy` payload |
| `nvml_backend.py` | Real-hardware 100 ms loop: NVML locked clocks and power limits; restores stock settings on exit. **Not run on real GPUs yet.** |
| `simulate.py` | Replays both headline claims |

```bash
cd case-studies/neuralwatt
python3 -m wattbudget.simulate          # ~4 s, stdlib only
python3 -m unittest discover -s tests   # 11 tests
```

### Simulated results (10 min, 100 ms ticks, sinusoidal demand, agent starts from a biased prior)

**1. Same 4,200 W envelope, demand above 6-GPU capacity at peaks**

| policy | GPUs | tokens served | peak tokens/s | tokens/J | peak node W | idle W/GPU | ticks over envelope |
|---|---|---|---|---|---|---|---|
| stock | 6 | 13.03 M | 28,478 | 5.23 | 4,194 | 125 | 0 |
| static 525 W cap | 8 | 14.40 M (+10.5%) | 30,627 (+7.5%) | 6.19 (+18.4%) | 4,134 | 104 | 0 |
| wattbudget | 8 | 14.34 M (+10.0%) | 30,490 (+7.1%) | **6.73 (+28.7%)** | 4,165 | **73** | 0 |

**2. Same 8 GPUs, no node cap, demand within capacity**

| policy | tokens served | tokens/J | idle W/GPU |
|---|---|---|---|
| stock | 11.40 M | 4.57 | 125 |
| wattbudget | 11.40 M (same work) | **5.96 (+30.4%)** | 83 |

### What this tells us

- **8 in the envelope of 6 is real physics, but most of it is just the
  cap.** A dumb static `nvidia-smi -pl 525` on 8 GPUs gets the same extra
  throughput as the smart agent. The agent's edge is efficiency: about +10%
  tokens/J over the static cap, mostly from idle downclocking and buying only
  the clock the queue needs.
- **We did not reproduce +33% throughput.** We got +7–10%, because during
  troughs the extra two GPUs have nothing to do. Neuralwatt's figure was
  measured at saturation. It may also include effects this model leaves out:
  thermals and leakage, CPU frequency tuning, and batch-size effects of
  spreading load over more GPUs.
- **Idle power is the cheapest win.** Simply not idling at boost clocks
  accounts for a large share of the tokens/J gain in both experiments.

## Limits and unknowns

- The simulator is a model. Every number above depends on its calibration,
  which was fit to public figures, not measured.
- `nvml_backend.py` has not been run on real GPUs. Clock-locking needs root,
  and H100 power limits bottom out around 200 W.
- Unknown: Neuralwatt's actual policy (the Q-learning state and reward), how
  it reads queue depth, CPU-side actions, and how the orchestrator
  coordinates multiple nodes (rack-level budgets, grid curtailment mode).
