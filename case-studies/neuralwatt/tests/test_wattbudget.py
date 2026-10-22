import json
import unittest
from pathlib import Path

from wattbudget.attribution import attribution_ratio, energy_payload
from wattbudget.budget import allocate, efficiency_ceiling
from wattbudget.gpu_model import DECODE, IDLE, PREFILL, GpuSpec, power, tps
from wattbudget.optimizer import CurveModel, Telemetry, classify
from wattbudget.simulate import BIASED_PRIOR, run

SPEC = GpuSpec()
EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "sse-energy-payload.json"


class GpuModelCalibration(unittest.TestCase):
    def test_matches_public_h100_figures(self):
        self.assertLessEqual(power(SPEC, SPEC.f_max, DECODE), SPEC.tdp)
        self.assertGreater(power(SPEC, SPEC.f_max, DECODE), 0.95 * SPEC.tdp)
        self.assertAlmostEqual(power(SPEC, SPEC.f_max, IDLE), 125, delta=1)  # stock idle on Crusoe
        self.assertLess(power(SPEC, SPEC.f_min, IDLE), 73)  # Neuralwatt idle on Crusoe

    def test_decode_is_clock_insensitive_prefill_is_not(self):
        half = SPEC.f_max / 2
        decode_kept = tps(SPEC, half, DECODE) / tps(SPEC, SPEC.f_max, DECODE)
        prefill_kept = tps(SPEC, half, PREFILL) / tps(SPEC, SPEC.f_max, PREFILL)
        self.assertGreater(decode_kept, 0.7)
        self.assertLess(prefill_kept, 0.6)


class Allocator(unittest.TestCase):
    def curve(self, phase):
        return [(f, power(SPEC, f, phase), tps(SPEC, f, phase)) for f in SPEC.levels()]

    def test_respects_envelope_and_feeds_busy_gpus(self):
        curves = [self.curve(DECODE)] * 3 + [self.curve(IDLE)]
        top = len(SPEC.levels()) - 1
        picks = allocate(1500.0, curves, [top] * 4)
        self.assertLessEqual(sum(c[i][1] for c, i in zip(curves, picks)), 1500.0)
        self.assertEqual(picks[3], 0)  # the idle GPU gets nothing
        self.assertGreater(min(picks[:3]), 0)

    def test_stops_at_demand(self):
        curves = [self.curve(DECODE)] * 4
        top = len(SPEC.levels()) - 1
        full = allocate(None, curves, [top] * 4)
        light = allocate(None, curves, [top] * 4, target_tps=4000.0)
        self.assertEqual(full, [top] * 4)
        self.assertLess(sum(light), sum(full))
        self.assertGreaterEqual(sum(c[i][2] for c, i in zip(curves, light)), 4000.0)

    def test_efficiency_ceiling_honours_slo(self):
        c = self.curve(DECODE)
        i = efficiency_ceiling(c, 0.05)
        self.assertGreaterEqual(c[i][2], 0.95 * c[-1][2])
        self.assertGreater(c[i][2] / c[i][1], c[-1][2] / c[-1][1])


class Learning(unittest.TestCase):
    def test_curve_model_converges_from_biased_prior(self):
        model = CurveModel(SPEC, BIASED_PRIOR)
        f = 1380
        truth = tps(SPEC, f, DECODE)
        before = abs(model.curve("decode")[SPEC.levels().index(f)][2] - truth)
        for _ in range(50):
            model.observe("decode", f, power(SPEC, f, DECODE), truth)
        after = abs(model.curve("decode")[SPEC.levels().index(f)][2] - truth)
        self.assertLess(after, 0.01 * before)

    def test_classify(self):
        self.assertEqual(classify(Telemetry(0.0, 0.0, 70, 0)), "idle")
        self.assertEqual(classify(Telemetry(0.5, 0.45, 300, 900)), "decode")
        self.assertEqual(classify(Telemetry(1.0, 0.35, 600, 9000)), "prefill")


class Attribution(unittest.TestCase):
    def test_reproduces_captured_neuralwatt_payload(self):
        observed = json.loads(EVIDENCE.read_text())["payload"]
        server_j = observed["avg_power_watts"] * observed["duration_seconds"]
        ratio = observed["energy_joules"] / server_j
        self.assertAlmostEqual(ratio, observed["attribution_ratio"], delta=0.00005)
        ours = energy_payload(
            server_joules=server_j,
            duration_s=observed["duration_seconds"],
            request_tokens=ratio,
            pool_tokens=1.0,
            n_gpus=8,
            cap=0.25,
            grid_gco2_per_kwh=observed["grid_carbon_intensity_gco2perkwhr"],
            grid_id=observed["grid_id"],
        )
        for key in ("energy_joules", "energy_kwh", "avg_power_watts", "carbon_g_co2eq"):
            self.assertAlmostEqual(ours[key] / observed[key], 1.0, delta=0.002, msg=key)
        self.assertEqual(ours["attribution_method"], observed["attribution_method"])

    def test_cap_limits_lone_request(self):
        self.assertEqual(attribution_ratio(100, 100, 0.25), 0.25)
        self.assertEqual(attribution_ratio(10, 100, 0.25), 0.1)


class Claims(unittest.TestCase):
    def test_more_gpus_in_same_envelope(self):
        envelope = 6 * SPEC.tdp
        stock = run("stock", 6, envelope, seconds=120)
        agent = run("wattbudget", 8, envelope, seconds=120)
        self.assertEqual(agent.over_envelope_ticks, 0)
        self.assertLessEqual(agent.peak_w, envelope)
        self.assertGreater(agent.tokens, 1.05 * stock.tokens)

    def test_fewer_joules_same_work(self):
        stock = run("stock", 8, None, seconds=120, mean_demand=19000, swing=0.45)
        agent = run("wattbudget", 8, None, seconds=120, mean_demand=19000, swing=0.45)
        self.assertLess(agent.backlog_tokens, 0.01 * agent.tokens)
        self.assertGreater(agent.tokens_per_joule, 1.15 * stock.tokens_per_joule)
        self.assertLess(agent.idle_w, 90)


if __name__ == "__main__":
    unittest.main()
