"""Tests for the six-variable program: closed form, alpha allocation, external comparators and config wiring."""
import itertools, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import binom
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from topic_experiment import engine as e, sixvar as s, violations as v, w3w2, runtime as r
from topic_experiment.common import stationary

DET = [.95, .2]


def moments_of_K(K):
    """Five moments and theta of a stationary pair law K on {0,1,2}^2 (rows Z_0, columns Z_1)."""
    pi = K.sum(1); m = pi[1] + pi[2]; y = pi[2]; h = K[2, 2]; a = K[2, 1] + K[2, 2]; b = K[1, 2] + K[2, 2]
    theta = 1 - 4*m + 4*K[1:, 1:].sum()
    return (m, y, a, b, h), theta


def random_pair_laws(rng, n):
    out = []
    for i in range(n):
        P = rng.dirichlet(np.ones(3)*rng.choice([.2, 1., 5.]), size=3)
        if i % 3 == 0:                       # sparse transitions put entries of K on the boundary
            P = P*(rng.random((3, 3)) < .6) + np.eye(3)*1e-3; P = P/P.sum(1, keepdims=True)
        out.append(stationary(P)[:, None]*P)
    return out


BOUNDARY = {
    'm=0': (0., 0., 0., 0., 0.), 'm=1,y=0': (1., 0., 0., 0., 0.), 'm=1,y=1': (1., 1., 1., 1., 1.),
    'y=0': (.4, 0., 0., 0., 0.), 'y=m': (.5, .5, .3, .3, .3), 'y=m, a=b=h=y': (.3, .3, .3, .3, .3),
    'a=b': (.5, .2875, .1, .1, .03), 'a=b=h': (.6, .3, .1, .1, .1), 'l=u': (.3, .2, .15, .15, .05),
    'L4 counterexample': (29/40, 23/60, 13/48, 77/240, 7/48),
}


class SixVariableProgram(unittest.TestCase):
    def test_lp_equals_closed_form_at_population_moments(self):
        rng = np.random.default_rng(20261013); worst = 0.
        cases = [moments_of_K(K) for K in random_pair_laws(rng, 300)]
        cases += [(tuple(v.analyst_law(v.make_condition('baseline', 0, st))[k] for k in s.NAMES), 2*st - 1) for st in (.2, .5, .8)]
        for mom, theta in cases:
            self.assertTrue(bool(s.compatible(*mom, tol=1e-12)))
            lo, hi = s.closed_form(*mom); plo, phi = s.population_interval(*mom)
            worst = max(worst, abs(plo - lo), abs(phi - hi))
            self.assertLessEqual(lo, theta + 1e-12); self.assertGreaterEqual(hi, theta - 1e-12)
        self.assertLess(worst, 1e-9)
        for name, mom in BOUNDARY.items():
            self.assertTrue(bool(s.compatible(*mom)), name)
            lo, hi = s.closed_form(*mom); plo, phi = s.population_interval(*mom)
            self.assertAlmostEqual(plo, float(lo), delta=1e-9, msg=name); self.assertAlmostEqual(phi, float(hi), delta=1e-9, msg=name)
        lo, hi = s.population_interval(*BOUNDARY['L4 counterexample'])
        self.assertAlmostEqual(lo, -1/10, delta=1e-9); self.assertAlmostEqual(hi, 11/20, delta=1e-9)

    def test_closed_form_matches_word_lp_population_bounds(self):
        for st in (.2, .5, .8):
            law = v.analyst_law(v.make_condition('baseline', 0, st)); lo, hi = s.closed_form(*[law[k] for k in s.NAMES])
            wl, wh = e.population_bounds(st, DET, 3, 'neighbor')
            self.assertAlmostEqual(float(lo), wl, delta=1e-6); self.assertAlmostEqual(float(hi), wh, delta=1e-6)
            # Population moments stated in the manuscript's simulation study: y=(.95(1-s)+.2s)/2, a=sy, b=.1s, h=.2sy.
            y = (.95*(1 - st) + .2*st)/2
            for k, want in zip(s.NAMES, (.5, y, st*y, .1*st, .2*st*y)): self.assertAlmostEqual(law[k], want, delta=1e-12)

    def test_boxes_containing_the_truth_give_intervals_containing_the_identified_interval(self):
        rng = np.random.default_rng(7)
        for K in random_pair_laws(rng, 150):
            mom, _ = moments_of_K(K); lo, hi = s.closed_form(*mom)
            for scale in (0., 1e-6, .01, .1, 1.):
                box = {k: (val - scale*rng.random(), val + scale*rng.random()) for k, val in zip(s.NAMES, mom)}
                blo, bhi = s.six_variable_program(box)
                self.assertLessEqual(blo, lo + 1e-9); self.assertGreaterEqual(bhi, hi - 1e-9)
                self.assertTrue(-1 - 1e-6 <= blo <= bhi + 1e-9 and bhi <= 1 + 1e-6)   # HiGHS feasibility tolerance (1e-7) times |c|=4
                if scale == 1e-6:            # shrinking boxes collapse to the identified set
                    self.assertLess(abs(blo - lo) + abs(bhi - hi), 1e-4)

    def test_nested_boxes_are_monotone(self):
        rng = np.random.default_rng(8)
        for K in random_pair_laws(rng, 60):
            mom, _ = moments_of_K(K)
            small = {k: (val - .01, val + .01) for k, val in zip(s.NAMES, mom)}; big = {k: (val - .05, val + .03) for k, val in zip(s.NAMES, mom)}
            a = s.six_variable_program(small); b = s.six_variable_program(big)
            self.assertLessEqual(b[0], a[0] + 1e-9); self.assertGreaterEqual(b[1], a[1] - 1e-9)

    def test_sample_interval_contains_identified_interval_whenever_the_region_holds(self):
        # Corollary: on the event that all five limits contain the true moments, containment is deterministic.
        rng = np.random.default_rng(11); held = 0
        for rep in range(300):
            st = (.2, .5, .8)[rep % 3]; law = v.analyst_law(v.make_condition('baseline', 0, st)); true = {k: law[k] for k in s.NAMES}
            _, y = e.sample_episodes(rng, 800, 3, st, DET); x, ay = e.sample_episodes(rng, 60, 3, st, DET)
            for interval in ('exact_binomial', 'hoeffding', 'empirical_bernstein'):
                lim = s.six_variable_limits(y, x[:, 1], ay, .2, interval)
                if all(lim[k][0] <= true[k] <= lim[k][1] for k in s.NAMES):
                    held += 1; lo, hi = s.six_variable_bounds(y, x[:, 1], ay, .2, interval); plo, phi = s.closed_form(*true.values())
                    self.assertLessEqual(lo, plo + 1e-9); self.assertGreaterEqual(hi, phi - 1e-9)
        self.assertGreater(held, 500)

    def test_empty_set_raises_and_runner_records_flagged_fallback(self):
        with self.assertRaises(RuntimeError):
            s.six_variable_program(dict(m=(.1, .2), y=(.5, .6), a=(0, 1), b=(0, 1), h=(0, 1)))      # y > m
        with self.assertRaises(RuntimeError):
            s.six_variable_program(dict(m=(.5, .5), y=(.3, .3), a=(.1, .1), b=(.1, .1), h=(.2, .2)))  # h > a
        # A limit overshooting [0,1] by rounding must not empty the box (both ends are clipped).
        lo, hi = s.six_variable_program(dict(m=(1 + 2e-16, 1 + 2e-16), y=(-1e-17, -1e-17), a=(0, 0), b=(0, 0), h=(0, 0))); self.assertAlmostEqual(lo, 1., delta=1e-9); self.assertAlmostEqual(hi, 1., delta=1e-9)
        y = np.ones((50, 3), int); ay = np.zeros((40, 3), int); ax = np.zeros(40, int)           # observed y=1 but audited m=0
        row = w3w2.evaluate('sixvar/exact_binomial', y, ax, ay, .05, [0, 0, 0], 0., pop=[-.1, .1])
        self.assertTrue(row['empty_confidence_set']); self.assertEqual((row['low'], row['high'], row['width']), (-1., 1., 2.))
        self.assertTrue(row['covered'] and row['contains_population'])


class AlphaAllocation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3); _, self.y = e.sample_episodes(rng, 2000, 3, .5, DET); x, self.ay = e.sample_episodes(rng, 200, 3, .5, DET); self.ax = x[:, 1]

    def test_six_variable_uses_exactly_five_intervals_at_alpha_over_five(self):
        calls = []; real = e.confidence_limits
        with patch.object(e, 'confidence_limits', side_effect=lambda vals, delta, interval: (calls.append((len(vals), delta, interval)), real(vals, delta, interval))[1]):
            s.six_variable_bounds(self.y, self.ax, self.ay, alpha=.05, interval='exact_binomial')
        self.assertEqual(len(calls), 5); self.assertTrue(all(abs(d - .01) < 1e-15 and i == 'exact_binomial' for _, d, i in calls))
        self.assertEqual(sorted(n for n, _, _ in calls), [200, 200, 200, 2000, 2000])   # m,a,b from audits; y,h from observations
        cols = s.five_moment_samples(self.y, self.ax, self.ay)
        np.testing.assert_array_equal(cols['m'], self.ax); np.testing.assert_array_equal(cols['a'], self.ay[:, 0]*self.ax)
        np.testing.assert_array_equal(cols['b'], self.ax*self.ay[:, 2]); np.testing.assert_array_equal(cols['y'], self.y[:, 1])
        np.testing.assert_array_equal(cols['h'], self.y[:, 0]*self.y[:, 1])
        lim = s.six_variable_limits(self.y, self.ax, self.ay, .05, 'exact_binomial')
        for k in s.NAMES: self.assertEqual(lim[k], real(cols[k], .05/5, 'exact_binomial'))

    def test_word_program_uses_eleven_intervals_at_alpha_over_eleven(self):
        calls = []; real = e.confidence_limits
        with patch.object(e, 'confidence_limits', side_effect=lambda vals, delta, interval: (calls.append(delta), real(vals, delta, interval))[1]):
            e.bounds(self.y, self.ax, self.ay, alpha=.05, moments='neighbor', interval='exact_binomial')
        self.assertEqual(len(calls), 11); self.assertTrue(all(abs(d - .05/11) < 1e-15 for d in calls))

    def test_every_constraint_of_the_program_is_either_one_of_five_limits_or_deterministic(self):
        # With degenerate (exact) boxes no sampling enters, so the only data-dependent constraints are the five boxes.
        mom = BOUNDARY['a=b']; base = s.population_interval(*mom)
        for i, k in enumerate(s.NAMES):
            box = {n: (val, val) for n, val in zip(s.NAMES, mom)}; box[k] = (mom[i] - .02, mom[i] + .02)
            lo, hi = s.six_variable_program(box); self.assertLessEqual(lo, base[0] + 1e-9); self.assertGreaterEqual(hi, base[1] - 1e-9)

    def test_union_of_five_cp_intervals_has_joint_coverage_by_exact_enumeration(self):
        # Exact per-feature miss probability of the alpha/5 CP interval is <= alpha/5 for every p, so the union bound gives 1-alpha.
        for n in (50, 500):
            k = np.arange(n + 1); lims = np.array([e.confidence_limits(np.r_[np.ones(i), np.zeros(n - i)], .01, 'exact_binomial') for i in k])
            for p in np.linspace(0, 1, 201):
                miss = binom.pmf(k, n, p)[(lims[:, 0] > p) | (lims[:, 1] < p)].sum(); self.assertLessEqual(miss, .01 + 1e-12)


class Comparators(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5); _, self.y = e.sample_episodes(rng, 20000, 3, .8, DET); x, self.ay = e.sample_episodes(rng, 700, 3, .8, DET); self.ax = x[:, 1]

    def test_cell_counts_reproduce_the_five_sample_moments(self):
        oc = s.obs_counts(self.y); ac = s.audit_counts5(self.ax, self.ay); a16 = s.audit_counts16(self.ax, self.ay)
        self.assertEqual(oc.sum(), 20000); self.assertEqual(ac.sum(), 700); self.assertEqual(a16.sum(), 700)
        for i, pat in enumerate(s.OBS_PATTERNS): self.assertEqual(oc[i], int(np.all(self.y == pat, axis=1).sum()))
        got = s.moments_from_cells(oc/oc.sum(), ac/ac.sum()); want = {k: val.mean() for k, val in s.five_moment_samples(self.y, self.ax, self.ay).items()}
        for g, k in zip(got, s.NAMES): self.assertAlmostEqual(float(g), want[k], delta=1e-12)
        self.assertEqual(a16[:8].sum(), ac[0]); self.assertEqual(a16.reshape(2, 2, 2, 2)[1, 1].sum(), ac[3] + ac[4])

    def test_l1_ball_radius_and_containment(self):
        self.assertAlmostEqual(s.l1_radius(8, 50000, .025), np.sqrt(2*np.log(254/.025)/50000), delta=1e-15)
        self.assertAlmostEqual(s.l1_radius(5, 500, .025), np.sqrt(2*np.log(30/.025)/500), delta=1e-15)
        for st in (.2, .5, .8):
            law = v.analyst_law(v.make_condition('baseline', 0, st)); plo, phi = s.closed_form(*[law[k] for k in s.NAMES])
            A = law['audit16'].reshape(2, 2, 2, 2); q5 = np.r_[A[0].sum(), [A[1, r0, :, s0].sum() for r0 in (0, 1) for s0 in (0, 1)]]
            for N, B in ((50000, 500), (10**12, 10**12)):
                lo, hi = s.l1_ball_bounds(law['obs8']*N, q5*B, .05)        # centred at the truth: the truth lies in both balls
                self.assertLessEqual(lo, plo + 1e-7); self.assertGreaterEqual(hi, phi - 1e-7)
            self.assertLess(abs(lo - plo) + abs(hi - phi), 1e-3)              # vanishing radii recover the identified interval
        lo, hi = s.l1_ball_bounds(s.obs_counts(self.y), s.audit_counts5(self.ax, self.ay), .05); self.assertTrue(-1 <= lo <= .6 <= hi <= 1)

    def test_l1_ball_region_contains_truth_implies_interval_contains_identified_set(self):
        rng = np.random.default_rng(21); held = 0; st = .5
        law = v.analyst_law(v.make_condition('baseline', 0, st)); plo, phi = s.closed_form(*[law[k] for k in s.NAMES])
        A = law['audit16'].reshape(2, 2, 2, 2); q5 = np.r_[A[0].sum(), [A[1, r0, :, s0].sum() for r0 in (0, 1) for s0 in (0, 1)]]
        for _ in range(60):
            oc = rng.multinomial(3000, law['obs8']); ac = rng.multinomial(200, q5)
            if abs(oc/3000 - law['obs8']).sum() <= s.l1_radius(8, 3000, .025) and abs(ac/200 - q5).sum() <= s.l1_radius(5, 200, .025):
                held += 1; lo, hi = s.l1_ball_bounds(oc, ac, .05); self.assertLessEqual(lo, plo + 1e-7); self.assertGreaterEqual(hi, phi - 1e-7)
        self.assertGreater(held, 50)

    def test_bootstrap_and_dirichlet_are_reproducible_and_centre_on_the_plug_in(self):
        oc = s.obs_counts(self.y); ac = s.audit_counts5(self.ax, self.ay); plug = s.closed_form(*s.moments_from_cells(oc/oc.sum(), ac/ac.sum()))
        for f in (s.bootstrap_interval, s.dirichlet_interval):
            a = f(oc, ac, .05, np.random.default_rng([1, 2])); b = f(oc, ac, .05, np.random.default_rng([1, 2])); self.assertEqual(a[:2], b[:2])
            self.assertLess(a[0], float(plug[0])); self.assertGreater(a[1], float(plug[1])); self.assertTrue(-1 <= a[0] <= a[1] <= 1)
            big = f(oc*10**6, ac*10**6, .05, np.random.default_rng(3))          # huge samples: the interval collapses to the plug-in endpoints
            self.assertAlmostEqual(big[0], float(plug[0]), delta=2e-3); self.assertAlmostEqual(big[1], float(plug[1]), delta=2e-3)
        info = s.dirichlet_interval(np.array([0, 0, 0, 0, 0, 0, 0, 50]), np.array([40, 0, 0, 0, 0]), .05, np.random.default_rng(0))[2]
        self.assertTrue(info['fallback']); self.assertGreater(info['discard_fraction'], .97)


class CoverageCheck(unittest.TestCase):
    def test_nominal_coverage_2000_seeds_no_violation(self):
        """Coverage check: stay=.5, B=500, N=50,000, seeds 70001-72000; all methods are evaluated on the same samples."""
        st = .5; law = v.analyst_law(v.make_condition('baseline', 0, st)); pop = [float(x) for x in s.closed_form(*[law[k] for k in s.NAMES])]
        methods = ['sixvar/exact_binomial', 'sixvar/hoeffding', 'sixvar/empirical_bernstein', 'l1ball', 'bootstrap', 'dirichlet']
        cover = dict.fromkeys(methods, 0); contain = dict.fromkeys(methods, 0); empty = dict.fromkeys(methods, 0); n = 2000
        for seed in range(70001, 70001 + n):
            rng = np.random.default_rng(seed); _, y = e.sample_episodes(rng, 50000, 3, st, DET); x, ay = e.sample_episodes(rng, 500, 3, st, DET)
            for mth in methods:
                row = w3w2.evaluate(mth, y, x[:, 1], ay, .05, [seed, 500, 500], 2*st - 1, pop)
                cover[mth] += row['covered']; contain[mth] += row['contains_population']; empty[mth] += row['empty_confidence_set']
        print('\n[coverage check, 2000 seeds] covered:', cover, 'contains identified interval:', contain, 'empty:', empty)
        for mth in methods[:4]:                                # procedures with a finite-sample guarantee
            self.assertGreaterEqual(cover[mth]/n, .95, mth); self.assertGreaterEqual(contain[mth]/n, .95, mth)
            self.assertEqual(empty[mth], 0, mth)


class Wiring(unittest.TestCase):
    def test_experiment_configs_validate_and_unknown_experiments_are_rejected(self):
        for name in ('w3', 'w2a', 'w2b'):
            for stage in ('smoke', 'main'):
                c = json.loads((ROOT/f'configs/{stage}_{name}.json').read_text()); self.assertEqual((c['stage'], c['experiment']), (stage, name)); r.validate_config(c)
                with self.assertRaisesRegex(ValueError, 'differs from the canonical'): r.validate_config(dict(c, alpha=.1))
                with self.assertRaisesRegex(ValueError, 'seed'): r.validate_config(dict(c, seeds=[1]))
        c = json.loads((ROOT/'configs/smoke_w3.json').read_text())
        with self.assertRaises((ValueError, OSError)): r.validate_config(dict(c, experiment='nope'))
        with self.assertRaisesRegex(ValueError, 'Invalid experiment'): r.validate_config(dict(c, experiment='../main'))
        plain = json.loads((ROOT/'configs/smoke.json').read_text())
        with self.assertRaises(ValueError): r.validate_config(dict(c, experiment=None))   # cannot fall back to configs/smoke.json
        with self.assertRaises(ValueError): r.validate_config(dict(plain, experiment='w3'))
        main = {n: set(json.loads((ROOT/f'configs/main_{n}.json').read_text())['seeds']) for n in ('w3', 'w2a', 'w2b')}
        self.assertEqual(main['w3'], set(range(92001, 93001))); self.assertEqual(main['w2a'], set(range(92001, 93001))); self.assertEqual(main['w2b'], set(range(93001, 93201)))
        old = set(json.loads((ROOT/'configs/main.json').read_text())['seeds'])
        smoke = set().union(*[set(json.loads((ROOT/f'configs/smoke_{n}.json').read_text())['seeds']) for n in ('w3', 'w2a', 'w2b')])
        self.assertFalse(old & (main['w3'] | main['w2b'])); self.assertFalse(smoke & (old | main['w3'] | main['w2b']))

    def test_w3_runner_rows_and_counts(self):
        c = dict(json.loads((ROOT/'configs/smoke_w3.json').read_text()), seeds=[21101], episodes=400, audit_budgets=[40])
        with tempfile.TemporaryDirectory() as td:
            rows = e.run(c, Path(td)); self.assertEqual(len(rows), 3*len(c['methods']))
            z = np.load(Path(td)/'counts.npz'); self.assertEqual(int(z['obs_21101_0.2'].sum()), 400); self.assertEqual(int(z['aud_21101_0.2_40'].sum()), 40)
            for row in rows:
                self.assertTrue(-1 - 1e-7 <= row['low'] and row['high'] <= 1 + 1e-7, row); self.assertEqual(row['experiment'], 'w3'); json.dumps(row, allow_nan=False)
            # The word-program rows equal a direct engine.bounds call on the same random stream.
            rng = np.random.default_rng(21101); _, y = e.sample_episodes(rng, 400, 3, .2, DET); x, ay = e.sample_episodes(rng, 40, 3, .2, DET)
            ref = e.bounds(y, x[:, 1], ay, moments='neighbor', interval='exact_binomial')
            got = [row for row in rows if row['stay'] == .2 and row['method'] == 'word/exact_binomial'][0]; self.assertEqual([got['low'], got['high']], ref)
            np.testing.assert_array_equal(z['obs_21101_0.2'], s.obs_counts(y))
        from topic_experiment.analysis import analyze
        out = analyze('C11', list(rows)); self.assertEqual(out['kind'], 'per_job_summary'); self.assertTrue(out['execution_quality']['claim_eligible'])

    def test_archived_word_rows_are_reproduced_exactly(self):
        runs = sorted(p for p in (ROOT/'results/runs/v4_2/main').glob('*/C11-generated-00[1-2]/attempt_*') if p.is_dir())
        if not runs or not list(runs[0].glob('*.npz')): self.skipTest('archived arrays not present')
        checked = 0
        for run in runs:
            rows = {(q['seed'], q['stay'], q['audits'], q['method']): q for q in json.loads((run/'rows.json').read_text()) if q['L'] == 3 and q['source'] == 'both_sampled'}
            for f in sorted(run.glob('*_3_*.npz')):
                seed, st, _, budget = f.stem.split('_'); z = np.load(f)
                for interval in ('hoeffding', 'empirical_bernstein', 'exact_binomial'):
                    old = rows[(int(seed), float(st), int(budget), f'neighbor/{interval}')]
                    new = w3w2.evaluate(f'word/{interval}', z['y'], z['audit_x'], z['audit_y'], .05, [0, 0, 0], old['truth'])
                    self.assertEqual((new['low'], new['high'], new['empty_confidence_set'], new['covered']), (old['low'], old['high'], old['empty_confidence_set'], old['covered'])); checked += 1
        self.assertEqual(checked, len(runs)*45)


class AdditionalRegressions(unittest.TestCase):
    """Regression tests that pin the runner, random streams and samplers against targeted code mutations."""
    def test_w3_runner_every_method_recomputed(self):
        c = dict(json.loads((ROOT/'configs/smoke_w3.json').read_text()), seeds=[21102], episodes=900, audit_budgets=[80, 200])
        with tempfile.TemporaryDirectory() as td: rows = e.run(c, Path(td))
        got = {(q['stay'], q['audits'], q['method']): q for q in rows}; rng = np.random.default_rng(21102)
        for st in c['stay']:
            law = v.analyst_law(v.make_condition('baseline', 0, st)); pop = [float(z) for z in s.closed_form(*[law[k] for k in s.NAMES])]
            _, y = e.sample_episodes(rng, 900, 3, st, DET)
            for B in (80, 200):
                x, ay = e.sample_episodes(rng, B, 3, st, DET); ax = x[:, 1]; oc = s.obs_counts(y); ac = s.audit_counts5(ax, ay); key = [21102, int(round(1000*st)), B]
                want = {f'word/{i}': e.bounds(y, ax, ay, alpha=.05, moments='neighbor', interval=i) for i in ('hoeffding', 'empirical_bernstein', 'exact_binomial')}
                want.update({f'sixvar/{i}': s.six_variable_bounds(y, ax, ay, .05, i) for i in ('hoeffding', 'empirical_bernstein', 'exact_binomial')})
                want['l1ball'] = s.l1_ball_bounds(oc, ac, .05)
                want['bootstrap'] = list(s.bootstrap_interval(oc, ac, .05, np.random.default_rng(key + [4]))[:2]); want['dirichlet'] = list(s.dirichlet_interval(oc, ac, .05, np.random.default_rng(key + [5]))[:2])
                for mth, (lo, hi) in want.items():
                    q = got[(st, B, mth)]; self.assertEqual([q['low'], q['high']], [lo, hi], (st, B, mth))
                    self.assertEqual(q['covered'], bool(lo <= 2*st - 1 <= hi)); self.assertEqual(q['contains_population'], bool(lo <= pop[0] + 1e-9 and hi >= pop[1] - 1e-9))
                    self.assertEqual([q['population_low'], q['population_high']], pop); self.assertEqual(q['guarantee'], mth not in ('bootstrap', 'dirichlet'))

    def test_bootstrap_and_dirichlet_definitions(self):
        oc = np.array([3000, 400, 500, 300, 450, 200, 100, 50]); ac = np.array([250, 120, 30, 60, 40]); N = oc.sum(); B = ac.sum()
        rng = np.random.default_rng(77); P = rng.multinomial(N, oc/N, size=2000)/N; Q = rng.multinomial(B, ac/B, size=2000)/B
        lo, hi = s.closed_form(*s.moments_from_cells(P, Q))
        got = s.bootstrap_interval(oc, ac, .05, np.random.default_rng(77)); self.assertEqual(got[:2], (max(-1., float(np.quantile(lo, .025))), min(1., float(np.quantile(hi, .975)))))
        rng = np.random.default_rng(78); P = rng.dirichlet(oc + .5, size=4000); Q = rng.dirichlet(ac + .5, size=4000); mom = s.moments_from_cells(P, Q); keep = s.compatible(*mom)
        lo, hi = s.closed_form(*[z[keep] for z in mom]); got = s.dirichlet_interval(oc, ac, .05, np.random.default_rng(78))
        self.assertEqual(got[:2], (max(-1., float(np.quantile(lo, .025))), min(1., float(np.quantile(hi, .975))))); self.assertEqual(got[2]['discard_fraction'], float(1 - keep.mean()))

    def test_l1_ball_against_explicit_sign_vector_formulation(self):
        from scipy.optimize import linprog
        rng = np.random.default_rng(31); _, y = e.sample_episodes(rng, 4000, 3, .2, DET); x, ay = e.sample_episodes(rng, 300, 3, .2, DET); ax = x[:, 1]
        words, X, Y, eq, beq, target = e._lp_setup(3)
        obs_f = np.array([np.all(Y == o, axis=1) for o in itertools.product(range(2), repeat=3)], float)
        aud_f = np.array([X[:, 1] == 0] + [(X[:, 1] == 1) & (Y[:, 0] == r0) & (Y[:, 2] == s0) for r0 in (0, 1) for s0 in (0, 1)], float)
        po = np.array([np.mean(np.all(y == o, axis=1)) for o in itertools.product(range(2), repeat=3)])
        pa = np.array([np.mean(ax == 0)] + [np.mean((ax == 1) & (ay[:, 0] == r0) & (ay[:, 2] == s0)) for r0 in (0, 1) for s0 in (0, 1)])
        A = []; b = []
        for F, c0, n in ((obs_f, po, 4000), (aud_f, pa, 300)):
            rad = np.sqrt(2*np.log((2**len(c0) - 2)/.025)/n)
            for sg in itertools.product((-1., 1.), repeat=len(c0)): A.append(np.array(sg)@F); b.append(rad + np.array(sg)@c0)     # sum |F p - c| <= rad
        ref = [float(target@linprog(sg*target, A_ub=np.array(A), b_ub=b, A_eq=eq, b_eq=beq, bounds=(0, None), method='highs').x) for sg in (1, -1)]
        got = s.l1_ball_bounds(s.obs_counts(y), s.audit_counts5(ax, ay), .05); np.testing.assert_allclose(got, ref, atol=1e-7)

    def test_near_degenerate_boxes_do_not_crash(self):
        for mom in ((1., .4, .4, .4, .1), (1., 1., 1., 1., 1.), (.5, .5, .3, .3, .3)):
            for eps in (1e-7, 1e-8, 1e-6):
                lo, hi = s.six_variable_program({k: (val - eps, val + eps) for k, val in zip(s.NAMES, mom)}); plo, phi = s.closed_form(*mom)
                self.assertLessEqual(lo, float(plo) + 1e-6); self.assertGreaterEqual(hi, float(phi) - 1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
