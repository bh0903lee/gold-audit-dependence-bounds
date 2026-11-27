"""Tests for the assumption-violation study: contaminated simulators, exact laws, breakdown bound and the refuting-audit event."""
import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from topic_experiment import engine as e, sixvar as s, violations as v, w3w2

DET = [.95, .2]
ZERO_LEVELS = [('V1', 0.), ('V2a', 'stay+0'), ('V2a', 'det+0'), ('V2b', 0.), ('V4a', 0.), ('V4b', 0.), ('V5', 1)]


def empirical(cond, seed, N, B):
    rng = np.random.default_rng(seed); y, audits = v.draw_condition(rng, cond, N, [B]); ax, ay, _ = audits[B]
    return s.obs_counts(y)/N, s.audit_counts16(ax, ay)/B


class Reduction(unittest.TestCase):
    def test_every_violation_axis_at_level_zero_is_the_baseline_stream(self):
        for stay in (.2, .5, .8):
            for seed in (1, 77):
                ref = np.random.default_rng([seed, 3]); _, y0 = e.sample_episodes(ref, 5000, 3, stay, DET)
                audits0 = [e.sample_episodes(ref, B, 3, stay, DET) for B in (70, 500)]
                for axis, level in ZERO_LEVELS + [('baseline', 0)]:
                    rng = np.random.default_rng([seed, 3]); y, audits = v.draw_condition(rng, v.make_condition(axis, level, stay), 5000, [70, 500])
                    np.testing.assert_array_equal(y, y0, err_msg=f'{axis} {level}')
                    for (x0, ay0), B in zip(audits0, (70, 500)):
                        ax, ay, xt = audits[B]
                        np.testing.assert_array_equal(ax, x0[:, 1]); np.testing.assert_array_equal(ay, ay0); np.testing.assert_array_equal(xt, x0[:, 1])
                        self.assertEqual(ax.dtype, x0.dtype); self.assertEqual(ay.dtype, ay0.dtype)
                    self.assertEqual(rng.bit_generator.state, ref.bit_generator.state, f'{axis} {level}: random stream consumed differently')

    def test_latent_sampler_equals_engine_sampler(self):
        a = np.random.default_rng(9); b = np.random.default_rng(9)
        for L in (3, 4):
            x0, y0 = e.sample_episodes(a, 2000, L, .8, DET); x1, y1 = v.sample_latent(b, 2000, L, v.markov1(.8))
            np.testing.assert_array_equal(x0, x1); np.testing.assert_array_equal(y0, y1)
        self.assertEqual(a.bit_generator.state, b.bit_generator.state)

    def test_level_zero_laws_equal_the_baseline_law_exactly(self):
        for stay in (.2, .5, .8):
            base = v.analyst_law(v.make_condition('baseline', 0, stay))
            np.testing.assert_allclose(base['obs8'], e.population_moments(stay, DET, 3, 'neighbor')[0], atol=1e-15)
            np.testing.assert_allclose([base['m'], base['a'], base['b']], e.population_moments(stay, DET, 3, 'neighbor')[1], atol=1e-15)
            self.assertAlmostEqual(base['truth'], 2*stay - 1, delta=1e-15)
            for axis, level in ZERO_LEVELS + [('V3a', 0.)]:
                law = v.analyst_law(v.make_condition(axis, level, stay))
                np.testing.assert_allclose(law['obs8'], base['obs8'], atol=1e-15); np.testing.assert_allclose(law['audit16'], base['audit16'], atol=1e-15)
                self.assertEqual(v.breakdown(v.make_condition(axis, level, stay), word_lp=False)['d'], 0.)

    def test_violations_change_the_data_only_where_intended(self):
        rng = np.random.default_rng(4); x, y = v.sample_latent(rng, 20000, 3, v.markov1(.5, fp=.1))
        self.assertTrue(np.any(y > x)); self.assertTrue(set(np.unique(y)) <= {0, 1})
        self.assertAlmostEqual(np.mean(y[x == 0]), .1, delta=.01)                       # false positives only where X=0
        x1 = np.r_[np.ones(50000, int), np.zeros(50000, int)]; y1 = np.tile([0, 1], 50000)
        flip = v.record_audit(np.random.default_rng(1), x1, y1, .05, 0.); self.assertAlmostEqual(np.mean(flip != x1), .05, delta=.004)
        anc = v.record_audit(np.random.default_rng(1), x1, y1, 0., .2); changed = anc != x1
        self.assertTrue(np.all((x1[changed] == 1) & (y1[changed] == 0))); self.assertAlmostEqual(changed[(x1 == 1) & (y1 == 0)].mean(), .2, delta=.01)
        for rho in (2, 4):
            xs, ys = v.sample_selected(np.random.default_rng(2), 60000, 3, v.markov1(.5), rho); self.assertEqual(len(xs), 60000)
            p = .2875; self.assertAlmostEqual(ys[:, 1].mean(), rho*p/(rho*p + 1 - p), delta=.008)
        self.assertTrue(np.all(ys <= xs))


class ExactLaws(unittest.TestCase):
    def test_exact_laws_are_probability_laws_with_stationary_pairs_inside_the_model(self):
        for stay in (.2, .5, .8):
            for axis, level in v.grid(list(v.LEVELS)):
                cond = v.make_condition(axis, level, stay); J = v.joint_law(cond['obs']); law = v.analyst_law(cond)
                self.assertAlmostEqual(J.sum(), 1., delta=1e-12); self.assertGreaterEqual(J.min(), 0.)
                self.assertAlmostEqual(law['obs8'].sum(), 1., delta=1e-12); self.assertAlmostEqual(law['audit16'].sum(), 1., delta=1e-12)
                self.assertAlmostEqual(law['truth'], law['truth_pair12'], delta=1e-12)      # latent X is stationary on every axis
                if axis in v.IN_MODEL:
                    self.assertEqual(float(J[0, :, :, 1].sum() + J[:, 0, :, :, 1].sum() + J[:, :, 0, :, :, 1].sum()), 0.)   # Y<=X
                    O = law['obs8'].reshape(2, 2, 2); np.testing.assert_allclose(O.sum(2), O.sum(0), atol=1e-12)          # stationary Y pairs
                    Z01 = J.sum(axis=(2, 5)); Z12 = J.sum(axis=(0, 3)); np.testing.assert_allclose(Z01, Z12, atol=1e-12)  # stationary (X,Y) pairs

    def test_simulators_match_exact_laws(self):
        # Every axis and level, one regime per condition; 6-sigma binomial bands on every cell (V6 uses dependent windows: wider band).
        N = 300000; B = 300000
        for i, (axis, level) in enumerate(v.grid(list(v.LEVELS))):
            stay = (.2, .5, .8)[i % 3]; cond = v.make_condition(axis, level, stay); law = v.analyst_law(cond)
            if axis == 'V6' and level == 1: n = 300000; width = 14.
            elif axis == 'V6': n = 100000 if level == 30 else 300000; width = 8.
            else: n = N; width = 6.
            obs, aud = empirical(cond, 1000 + i, n, n)
            for got, want, size in ((obs, law['obs8'], n), (aud, law['audit16'], n)):
                band = width*np.sqrt(np.maximum(want*(1 - want), 1e-12)/size) + 1e-9
                self.assertTrue(np.all(abs(got - want) <= band), f'{axis} {level} stay={stay}: max dev {abs(got - want).max():.5f}')
                self.assertTrue(np.all(got[want == 0] == 0))

    def test_v3a_truth_formula_and_v3_stay_inside_the_theorem(self):
        for stay in (.2, .5, .8):
            for g in v.LEVELS['V3a']:
                pss = stay + g*(1 - stay); pcs = stay - g*stay; q = pcs/(1 - pss + pcs)
                self.assertAlmostEqual(v.analyst_law(v.make_condition('V3a', g, stay))['truth'], 2*q - 1, delta=1e-12)
            for axis in ('V3a', 'V3b'):
                for level in v.LEVELS[axis]:
                    b = v.breakdown(v.make_condition(axis, level, stay))
                    self.assertTrue(b['compatible'] and b['word_feasible']); self.assertLess(b['d'], 1e-9); self.assertLess(b['word_d'], 1e-6)
                    self.assertAlmostEqual(b['word_low'], b['theta_low'], delta=1e-6); self.assertAlmostEqual(b['word_high'], b['theta_high'], delta=1e-6)
                    self.assertEqual(b['t1_probability'], 0.)


class Breakdown(unittest.TestCase):
    def test_closed_form_breakdown_matches_brute_force_monte_carlo(self):
        # Brute force: simulate, estimate the five analyst moments AND the truth from simulated latent pairs, apply the theorem's formulas.
        cases = [('V1', .10, .8), ('V1', .05, .8), ('V2a', 'stay+0.1', .2), ('V4a', .05, .2), ('V4b', .2, .8), ('V5', 2, .2)]
        n = 2_000_000
        for i, (axis, level, stay) in enumerate(cases):
            cond = v.make_condition(axis, level, stay); exact = v.breakdown(cond, word_lp=False); self.assertGreater(exact['d'], 1e-3)
            rng = np.random.default_rng(500 + i)
            x, y = v.sample_latent(rng, n, 3, cond['obs']); theta = float(np.mean((2*x[:, 0] - 1)*(2*x[:, 1] - 1)))
            xa, ya = v.sample_selected(rng, n, 3, cond['aud'], cond['rho']); rec = v.record_audit(rng, xa[:, 1], ya[:, 1], cond['flip'], cond['anchor'])
            mom = {k: float(val.mean()) for k, val in s.five_moment_samples(y, rec, ya).items()}
            lo, hi = (float(z) for z in s.closed_form(*[mom[k] for k in s.NAMES])); d = max(lo - theta, theta - hi, 0.)
            # Each of lo, hi, theta is a combination of <=5 Bernoulli means with coefficients <=4: sd <= 5*4*0.5/sqrt(n); use 5 sd.
            tol = 5*5*4*.5/np.sqrt(n)
            for k in s.NAMES: self.assertAlmostEqual(mom[k], exact[k], delta=5*.5/np.sqrt(n), msg=f'{axis} {level} {k}')
            self.assertAlmostEqual(theta, exact['truth'], delta=5*2/np.sqrt(n)); self.assertAlmostEqual(d, exact['d'], delta=tol, msg=f'{axis} {level}')
            self.assertGreater(d, 0.)

    def test_breakdown_is_zero_without_violation_and_definition_is_respected(self):
        for stay in (.2, .5, .8):
            b = v.breakdown(v.make_condition('baseline', 0, stay)); self.assertEqual(b['d'], 0.); self.assertTrue(b['compatible'])
            self.assertTrue(b['theta_low'] <= b['truth'] <= b['theta_high'])
        b = v.breakdown(v.make_condition('V1', .10, .8)); self.assertAlmostEqual(b['d'], max(b['theta_low'] - b['truth'], b['truth'] - b['theta_high'], 0.), delta=0)
        self.assertAlmostEqual(b['d'], .6 - b['theta_high'], delta=1e-15)

    def test_t1_probability(self):
        # Detection power of the refuting-audit event: 1-(1-phi(1-m))^B with m=.5, B=500 -> 39%, 71%, 92%, 99.4%.
        for phi, power in ((.002, .39), (.005, .71), (.01, .92), (.02, .994)):
            p = v.analyst_law(v.make_condition('V1', phi, .5))['t1_probability']; self.assertAlmostEqual(p, phi/2, delta=1e-12)
            self.assertAlmostEqual(1 - (1 - p)**500, power, delta=.006)
        for stay in (.2, .5, .8):
            y = v.analyst_law(v.make_condition('baseline', 0, stay))['y']
            self.assertAlmostEqual(v.analyst_law(v.make_condition('V4a', .02, stay))['t1_probability'], .02*y, delta=1e-12)
            for axis, level in (('V2a', 'stay+0.2'), ('V2b', .2), ('V4b', .2), ('V5', 4), ('V6', 1)):
                self.assertEqual(v.analyst_law(v.make_condition(axis, level, stay))['t1_probability'], 0.)


class Runner(unittest.TestCase):
    def test_w2_runner_rows_t1_and_baseline_identity(self):
        c = dict(json.loads((ROOT/'configs/smoke_w2b.json').read_text()), seeds=[21301], episodes=600, audit_budgets=[80], stay=[.5])
        with tempfile.TemporaryDirectory() as td:
            rows = e.run(c, Path(td)); conditions = v.grid(c['axes']); self.assertEqual(len(rows), len(conditions)*len(c['methods']))
            self.assertEqual({(q['axis'], q['level']) for q in rows}, {(a, str(l)) for a, l in conditions})
            z = np.load(Path(td)/'counts.npz')
            base = [q for q in rows if q['axis'] == 'baseline' and q['method'] == 'word/exact_binomial'][0]
            rng = np.random.default_rng([21301, 0, w3w2.W2_STREAM_TAG]); _, y = e.sample_episodes(rng, 600, 3, .5, DET); x, ay = e.sample_episodes(rng, 80, 3, .5, DET)
            self.assertEqual([base['low'], base['high']], e.bounds(y, x[:, 1], ay, moments='neighbor', interval='exact_binomial'))
            self.assertEqual(base['t1_count'], 0); np.testing.assert_array_equal(z['aud_21301_0.5_baseline_0_80'], s.audit_counts16(x[:, 1], ay))
            for q in rows:
                A = z[f"aud_21301_0.5_{q['axis']}_{q['level']}_80"].reshape(2, 2, 2, 2); self.assertEqual(q['t1_count'], int(A[0, :, 1, :].sum()))
                if q['axis'] in ('baseline', 'V2a', 'V2b', 'V3a', 'V3b', 'V4b', 'V5', 'V6'): self.assertEqual(q['t1_count'], 0)
                json.dumps(q, allow_nan=False)
            self.assertGreater(max(q['t1_count'] for q in rows if q['axis'] == 'V1' and q['level'] == '0.1'), 0)


class AdditionalRegressions(unittest.TestCase):
    """Regression tests that pin the runner, random streams and samplers against targeted code mutations."""
    def test_w2_stream_is_distinct_from_w3_and_restarted_per_condition(self):
        a = np.random.default_rng(4242).random(5); b = np.random.default_rng([4242, 0, w3w2.W2_STREAM_TAG]).random(5); self.assertFalse(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a, np.random.default_rng([4242, 0]).random(5)))      # the collision the tag avoids

    def test_w2_runner_rows_recomputed_from_recorded_labels(self):
        c = dict(json.loads((ROOT/'configs/smoke_w2b.json').read_text()), seeds=[21302], episodes=700, audit_budgets=[60, 150], stay=[.2, .8])
        with tempfile.TemporaryDirectory() as td:
            rows = e.run(c, Path(td))
        got = {(q['stay'], q['axis'], q['level'], q['audits'], q['method']): q for q in rows}; self.assertEqual(len(got), len(rows)); differs = 0
        for si, stay in enumerate(c['stay']):
            for axis, level in v.grid(c['axes']):
                cond = v.make_condition(axis, level, stay); truth = v.analyst_law(cond)['truth']
                y, audits = v.draw_condition(np.random.default_rng([21302, si, 2]), cond, 700, [60, 150])
                for B in (60, 150):
                    rec, ay, true_x = audits[B]; differs += int(np.any(rec != true_x))
                    for method, f in (('word/exact_binomial', lambda: e.bounds(y, rec, ay, alpha=.05, moments='neighbor', interval='exact_binomial')),
                                      ('word/hoeffding', lambda: e.bounds(y, rec, ay, alpha=.05, moments='neighbor', interval='hoeffding')),
                                      ('sixvar/exact_binomial', lambda: s.six_variable_bounds(y, rec, ay, .05, 'exact_binomial'))):
                        q = got[(stay, axis, str(level), B, method)]
                        try: lo, hi = f(); want = (lo, hi, False, bool(lo <= truth <= hi))
                        except e.InfeasibleConfidenceSet: want = (-1., 1., True, True)
                        self.assertEqual((q['low'], q['high'], q['empty_confidence_set'], q['covered']), want, (stay, axis, level, B, method))
                        self.assertEqual(q['truth'], truth); self.assertEqual(q['t1_count'], int(np.sum((rec == 0) & (ay[:, 1] == 1))))
        self.assertGreater(differs, 0)          # recorded and true labels do differ somewhere, so the check is not vacuous

    def test_covered_flag_uses_both_ends(self):
        y = np.random.default_rng(0); _, yy = e.sample_episodes(y, 3000, 3, .5, DET); x, ay = e.sample_episodes(y, 300, 3, .5, DET)
        lo, hi = s.six_variable_bounds(yy, x[:, 1], ay)
        for truth, want in ((lo - .01, False), (hi + .01, False), ((lo + hi)/2, True)):
            self.assertEqual(w3w2.evaluate('sixvar/exact_binomial', yy, x[:, 1], ay, .05, [0, 0, 0], truth)['covered'], want)
        for pop, want in (([lo - .01, hi - .01], False), ([lo + .01, hi + .01], False), ([lo + .01, hi - .01], True)):
            self.assertEqual(w3w2.evaluate('sixvar/exact_binomial', yy, x[:, 1], ay, .05, [0, 0, 0], 0., pop)['contains_population'], want)
        for interval in ('hoeffding', 'empirical_bernstein', 'exact_binomial'):
            r1 = w3w2.evaluate(f'sixvar/{interval}', yy, x[:, 1], ay, .05, [0, 0, 0], 0.); self.assertEqual([r1['low'], r1['high']], s.six_variable_bounds(yy, x[:, 1], ay, .05, interval))
        self.assertNotEqual(s.six_variable_bounds(yy, x[:, 1], ay, .05, 'hoeffding'), s.six_variable_bounds(yy, x[:, 1], ay, .05, 'exact_binomial'))

    def test_truth_matches_simulated_latent_pairs_on_every_axis(self):
        n = 400000
        for i, (axis, level) in enumerate(v.grid(list(v.LEVELS))):
            stay = (.8, .2, .5)[i % 3]; cond = v.make_condition(axis, level, stay); rng = np.random.default_rng(900 + i)
            if cond['step'] is not None: x, _ = v.sample_series_windows(rng, n, 3, stay, DET, cond['step']); tol = 12/np.sqrt(n) if cond['step'] == 1 else 6/np.sqrt(n)
            else: x, _ = v.sample_latent(rng, n, 3, cond['obs']); tol = 5/np.sqrt(n)
            truth = v.analyst_law(cond)['truth']
            self.assertAlmostEqual(float(np.mean((2*x[:, 0] - 1)*(2*x[:, 1] - 1))), truth, delta=tol, msg=f'{axis} {level}')
            self.assertAlmostEqual(float(np.mean((2*x[:, 1] - 1)*(2*x[:, 2] - 1))), truth, delta=tol, msg=f'{axis} {level}')

    def test_v6_windows_spacing_and_stationary_start(self):
        for step in (1, 3, 30):
            x, y = v.sample_series_windows(np.random.default_rng(step), 2000, 3, .8, DET, step); self.assertEqual(x.shape, (2000, 3)); self.assertTrue(np.all(y <= x))
            if step == 1: np.testing.assert_array_equal(x[1:, :2], x[:-1, 1:]); np.testing.assert_array_equal(y[1:, :2], y[:-1, 1:])
        # spacing: P(X at window k+1 start == X at window k end) = P(stay over step-2 steps) = (1+(2s-1)^(step-2))/2
        x, _ = v.sample_series_windows(np.random.default_rng(5), 400000, 3, .8, DET, 3); self.assertAlmostEqual(np.mean(x[1:, 0] == x[:-1, 2]), .8, delta=.004)
        x, _ = v.sample_series_windows(np.random.default_rng(6), 200000, 3, .8, DET, 30); self.assertAlmostEqual(np.mean(x[1:, 0] == x[:-1, 2]), (1 + .6**28)/2, delta=.006)
        # stationary start: the first window of fresh series follows the population law
        first = np.array([v.sample_series_windows(np.random.default_rng([8, i]), 1, 3, .8, DET, 1)[1][0] for i in range(40000)])
        want = v.analyst_law(v.make_condition('baseline', 0, .8))['obs8']; got = s.obs_counts(first)/40000
        self.assertTrue(np.all(abs(got - want) <= 5*np.sqrt(want*(1 - want)/40000) + 1e-9))

    def test_v2a_detector_drift_moves_both_components_and_only_the_audit_cohort(self):
        c = v.make_condition('V2a', 'det+0.1', .5); self.assertEqual(c['aud']['detect'], [.95 - .1, .2 + .1]); self.assertEqual(c['obs']['detect'], [.95, .2])
        c = v.make_condition('V2a', 'stay+0.1', .5); self.assertAlmostEqual(c['aud']['stay'], .6); self.assertEqual(c['obs']['stay'], .5); self.assertEqual(c['aud']['detect'], [.95, .2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
