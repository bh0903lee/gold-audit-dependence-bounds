"""Checks of the recorded-data procedures on cases with a known answer; no recorded data are read."""
import itertools, unittest
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta
import cohort_study as cs
from topic_experiment.sixvar import closed_form


def one_sided_law(m, rho, d0, d1):
    """Window law of a stationary two-state chain with one-sided detection."""
    a, b = m + rho * (1 - m), m * (1 - rho); T = np.array([[1 - b, b], [1 - a, a]]); det = (d0, d1); p = np.zeros(cs.NW)
    for i in range(cs.NW):
        xs, ys = cs.WX[i], cs.WY[i]
        if np.any((xs == 0) & (ys == 1)): continue
        for pv in (0, 1):
            q = m if pv else 1 - m; prev = pv
            for x, y in zip(xs, ys):
                q *= T[prev, x] * ((det[prev] if y else 1 - det[prev]) if x else 1.); prev = x
            p[i] += q
    return p


def draw(p, n, rng):
    i = rng.choice(cs.NW, size=n, p=p); return cs.WX[i], cs.WY[i]


class Population(unittest.TestCase):
    def test_neighbor_equals_full_equals_closed_form(self):
        rng = np.random.default_rng(1)
        for _ in range(25):
            p = one_sided_law(rng.uniform(.1, .7), rng.uniform(0, .9), rng.uniform(.1, .95), rng.uniform(.1, .95)); self.assertAlmostEqual(p.sum(), 1, places=12); s = cs.sharp(p)
            y1, y0, y2, x1 = cs.WY[:, 1], cs.WY[:, 0], cs.WY[:, 2], cs.WX[:, 1]; lo, hi = closed_form(p @ x1, p @ y1, p @ (y0 * x1), p @ (x1 * y2), p @ (y0 * y1))
            for k in ('theta_lower', 'theta_upper'): self.assertAlmostEqual(s['N'][k], s['F'][k], places=6)
            self.assertAlmostEqual(s['N']['theta_lower'], float(lo), places=6); self.assertAlmostEqual(s['N']['theta_upper'], float(hi), places=6)
            self.assertLessEqual(s['N']['width'], s['M']['width'] + 1e-7)

    def test_projection_keeps_a_stationary_law(self):
        p = one_sided_law(.3, .6, .5, .7)
        for f in (cs.project_ls, cs.project_tv): self.assertLess(np.abs(f(p) - p).max(), 1e-6)

    def test_projection_does_not_add_false_positives(self):
        rng = np.random.default_rng(11); x, y = draw(one_sided_law(.25, .5, .5, .6), 3000, rng); q = cs.window_law(x, y); p = cs.project_ls(q)
        self.assertEqual(float(cs.FP_ALL.sum(0) @ q), 0.); self.assertLess(float(cs.FP_ALL.sum(0) @ p), 1e-9); self.assertLess(np.abs(cs.EQ @ p - cs.EB).max(), 1e-9)
        s = cs.sharp(p); self.assertAlmostEqual(s['N']['width'], s['F']['width'], places=6)                      # Theorem 1 at a one-sided stationary law
        noisy = q.copy(); i = int(np.flatnonzero(cs.FP_ROW)[0]); noisy[i] += .01; noisy /= noisy.sum(); cap = float((cs.FP_ALL @ noisy).mean())
        self.assertLessEqual(float(cs.FP_ROW @ cs.project_ls(noisy)), cap + 1e-9)

    def test_scores_report_the_error_of_point_estimates(self):
        s = cs.score([{'a': (.1, .5), 'point estimates': {'a': .2}}, {'a': (.2, .6), 'point estimates': {'a': .5}}, {'a': (.3, .7), 'point estimates': {'a': .6}}], .3)['a']
        self.assertAlmostEqual(s['median_estimate'], .5); self.assertAlmostEqual(s['median_absolute_error'], .2)

    def test_equivalent_laws_share_the_data_law(self):
        p = one_sided_law(.4, .5, .6, .3); a, b, ca, cb, gap = cs.equivalent_laws(p); s = cs.sharp(p)
        self.assertLess(gap, 1e-8); self.assertAlmostEqual(ca, s['F']['lower'], places=6); self.assertAlmostEqual(cb, s['F']['upper'], places=6); self.assertGreater(cb - ca, .05)
        for q in (a, b): self.assertLess(np.abs(cs.EQ @ q - cs.EB).max(), 1e-8)


class Intervals(unittest.TestCase):
    def test_joint_correlation_range_matches_a_separate_program(self):
        rng = np.random.default_rng(2); x, y = draw(one_sided_law(.25, .6, .5, .5), 3000, rng); au = rng.permutation(3000)[:400]; ax, ay = x[au, 1], y[au]
        ST = [(0, 0), (1, 0), (1, 1)]; W = list(itertools.product(ST, repeat=3)); fx = lambda f: np.array([f(w) for w in W], float); al = .05 / 11; A, b = [], []
        cp = lambda k, n: (0. if k == 0 else float(beta.ppf(al / 2, k, n - k + 1)), 1. if k == n else float(beta.ppf(1 - al / 2, k + 1, n - k)))
        for t in itertools.product((0, 1), repeat=3):
            lo, hi = cp(int(np.all(y == t, axis=1).sum()), len(y)); f = fx(lambda w: all(w[i][1] == t[i] for i in range(3))); A += [f, -f]; b += [hi, -lo]
        for fn, k in ((lambda w: w[1][0], int(ax.sum())), (lambda w: w[1][0] * w[0][1], int((ax * ay[:, 0]).sum())), (lambda w: w[1][0] * w[2][1], int((ax * ay[:, 2]).sum()))):
            lo, hi = cp(k, len(ax)); f = fx(fn); A += [f, -f]; b += [hi, -lo]
            if fn(((0, 0), (1, 0), (0, 0))) == 1 and fn(((1, 1), (1, 0), (1, 1))) == 1 and fn(((0, 0), (1, 0), (1, 1))) == 1: ml, mh, fm = lo, hi, f
        E = [np.ones(27)]; e = [1.]
        for u, v in itertools.product(ST, repeat=2): E.append(fx(lambda w: (w[0] == u and w[1] == v) - (w[1] == u and w[2] == v))); e.append(0.)
        c = fx(lambda w: (2 * w[0][0] - 1) * (2 * w[1][0] - 1)); vals = []
        for m in np.linspace(max(ml, 1e-3), mh, 25):
            for s in (1, -1):
                r = linprog(s * c, A_ub=np.array(A), b_ub=b, A_eq=np.array(E + [fm]), b_eq=e + [m], bounds=(0, None), method='highs')
                if r.status == 0: vals.append(cs.corr_of(s * r.fun, m))
        got = cs.finite_sample(y, ax, ay, 0., .05, 'N'); lo, hi = max(min(vals), -1), min(max(vals), 1)                      # the outer range contains the range on a grid of points and is close to it
        self.assertLessEqual(got[0], lo + 1e-9); self.assertGreaterEqual(got[1], hi - 1e-9); self.assertLess(lo - got[0], .05); self.assertLess(got[1] - hi, .05)

    def test_plug_in_recovers_the_sharp_interval_of_a_large_sample(self):
        p = one_sided_law(.3, .5, .6, .4); rng = np.random.default_rng(3); x, y = draw(p, 400000, rng); s = cs.sharp(p)
        for k in cs.INFOS:
            lo, hi = cs.plug_in(y, x[:, 1], y, k); self.assertAlmostEqual(lo, s[k]['lower'], delta=.03); self.assertAlmostEqual(hi, s[k]['upper'], delta=.03)

    def test_imbens_manski_critical_values(self):
        lo, hi = cs.imbens_manski(0., 0., .1, .1); self.assertAlmostEqual(hi, .1 * 1.959964, places=4)          # point identified: two-sided
        lo, hi = cs.imbens_manski(0., .9, .01, .01); self.assertAlmostEqual(hi - .9, .01 * 1.644854, places=4)   # wide set: one-sided

    def test_empty_confidence_set_is_reported_as_none(self):
        rng = np.random.default_rng(4); x, y = draw(one_sided_law(.3, .5, .9, .9), 5000, rng); bad = np.zeros(300, int)      # audits that contradict the record
        self.assertIsNone(cs.finite_sample(y, bad, np.ones((300, 3), int), 0., .05, 'N'))


class Models(unittest.TestCase):
    def test_hidden_markov_intervals_cover_their_own_model(self):
        rng = np.random.default_rng(5); hits = {k: 0 for k in ('HMM, one-sided error', 'HMM, detection depends on previous state')}; truth = .63
        for _ in range(20):
            x, y = draw(one_sided_law(.3, truth, .5, .5), 6000, rng); out, est = cs.model_intervals(y, x[:400, 1], np.arange(400), np.arange(6000), rng, boots=(100, 2)); self.assertLess(abs(est['HMM, one-sided error'] - truth), .2)
            for k in hits: lo, hi = out[k]; hits[k] += lo <= truth <= hi; self.assertLess(hi - lo, .5)
        for k in hits: self.assertGreaterEqual(hits[k], 16, k)

    def test_chain_parameters(self):
        m, rho = cs.chain(.12, .82); self.assertAlmostEqual(m, .12 / (1 - .82 + .12)); self.assertAlmostEqual(rho, .7)
        np.testing.assert_allclose(cs.cell_probabilities(m, rho, .5, .5, 0.), cs.class_cells([.4, .12, .82, .12, .82, .5, .5, 0.]), atol=1e-12)

    def test_cell_probabilities_sum_to_one_and_match_the_window_law(self):
        c = cs.cell_probabilities(.3, .6, .5, .7, 0.); self.assertAlmostEqual(c.sum(), 1, places=12); p = one_sided_law(.3, .6, .5, .7)
        for j in range(8):
            for x1 in (0, 1): self.assertAlmostEqual(c[x1, j], p[(cs.WX[:, 1] == x1) & (cs.WY @ cs.POW == j)].sum(), places=12)

    def test_pattern_correlation(self):
        rng = np.random.default_rng(6); y = (rng.random((2000, 3)) < .3).astype(int); c = np.bincount(y @ cs.POW, minlength=8).astype(float)
        self.assertAlmostEqual(float(cs.pattern_corr(c)), cs.pooled(y), places=10)



def series(states, records, unit='u', extra=None):
    """Visits of one series on consecutive days, in the format of the loaders."""
    import datetime as dt
    return [dict(pid=unit, hh=unit, district=None, date=dt.date(2000, 1, 1) + dt.timedelta(days=i), age=0., y=int(r), q=int(s), density=1., conf=float(r), **(extra or {})) for i, (s, r) in enumerate(zip(states, records))]


class Windows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8); self.x = {u: (rng.random(40) < .5).astype(int) for u in 'abc'}; self.y = {u: v * (rng.random(40) < .6) for u, v in self.x.items()}
        self.data = ({u: series(self.x[u], self.y[u], u) for u in 'abc'}, {})

    def test_windows_are_consecutive_and_do_not_overlap(self):
        for lag, count in ((1, 13), (2, 8)):
            p = cs.build(self.data, cs._c(gap=(1, 1), lag=lag)); self.assertEqual(len(p['x']), 3 * count); self.assertEqual(p['hh'].max(), 2)
            np.testing.assert_array_equal(p['x'][:count], np.array([self.x['a'][s:s + 2 * lag + 1:lag] for s in range(0, 40 - 2 * lag, 2 * lag + 1)][:count]))
            np.testing.assert_array_equal(p['y'][:count], np.array([self.y['a'][s:s + 2 * lag + 1:lag] for s in range(0, 40 - 2 * lag, 2 * lag + 1)][:count]))

    def test_a_missing_occasion_breaks_the_window(self):
        d = ({'a': [v for i, v in enumerate(self.data[0]['a']) if i != 4]}, {}); p = cs.build(d, cs._c(gap=(1, 1)))
        np.testing.assert_array_equal(p['x'][:2], [self.x['a'][0:3], self.x['a'][5:8]])

    def test_one_window_per_unit_and_threshold(self):
        p = cs.build(self.data, cs._c(gap=(1, 1), sampling='participant')); self.assertEqual(len(p['x']), 3)
        hi = cs.build(self.data, cs._c(gap=(1, 1), threshold=2.)); self.assertEqual(hi['y'].sum(), 0)

    def test_long_series_are_not_truncated(self):
        n = 40000; p = cs.build(({'a': series(np.zeros(n), np.zeros(n))}, {}), cs._c(gap=(1, 1))); self.assertEqual(len(p['x']), n // 3)

    def test_resampling_draws_whole_units(self):
        p = cs.build(self.data, cs._c(gap=(1, 1))); x, y, hh = cs.resample(p, np.random.default_rng(1)); self.assertEqual(len(np.unique(hh)), 3)
        for h in range(3): self.assertEqual(int((hh == h).sum()), 13)


class Scores(unittest.TestCase):
    def test_empty_sets_fail_and_do_not_enter_the_width(self):
        s = cs.score([{'a': (.1, .5)}, {'a': None}, {'a': (.4, .6)}, None], .3)['a']
        self.assertAlmostEqual(s['coverage'], 1 / 3); self.assertAlmostEqual(s['empty_rate'], 1 / 3); self.assertAlmostEqual(s['width'], .3); self.assertAlmostEqual(s['positive'], 2 / 3)

    def test_unit_bootstrap_is_wider_when_units_differ(self):
        rng = np.random.default_rng(3); hh = np.repeat(np.arange(20), 100); level = np.repeat(rng.random(20) < .5, 100); a = (rng.random(2000) < np.where(level, .8, .2)).astype(int); b = (rng.random(2000) < np.where(level, .8, .2)).astype(int)
        lo, hi = cs.pair_interval(a, b, hh, rng); lo0, hi0 = cs.pair_interval(a, b, np.arange(2000), rng); self.assertGreater(hi - lo, 1.2 * (hi0 - lo0))
        c = cs.cluster_counts(2 * a + b, hh, 4, rng, 50); np.testing.assert_array_equal(c.sum(1), 2000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
