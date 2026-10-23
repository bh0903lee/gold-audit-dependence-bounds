"""Sensitivity of the recorded-occupancy reference intervals (Section 8.4) to the time stratum and the lag (2026-10-22).

Uses the same 60-second grid (derived_60s/occupancy_60s.csv, made by resample_occupancy_60s.py), the same window rule (three
instants at exact multiples of the lag within one calendar day), the same encoding X = (count > 0), Y = (S6_PIR or S7_PIR),
and the same stationary four-state reference program with the record's own false-positive mass as the budget
(addendum_core.population_references). Only the stratum and the lag vary. Also reports the number of windows whose first
two positions differ (the record's target is 1 - 2 * changes / n) and the smallest number of changes that the upper
endpoint allows.

Finite-sample part: as in Section 8.4, each replicate draws B audit and B calibration windows without replacement, sets the
false-positive budget to the one-sided Clopper-Pearson upper limit (0.99) on the calibration windows, and solves the
four-state program with simultaneous CP moment intervals (alpha 0.04); reported are containment of the record's target and
mean widths for the marginal (M), neighbor (N) and full (F) audits, with empty confidence sets (infeasibility certified by the
solver) and solver failures counted separately. Seeds 470001.. (one stream per replicate).

Usage (package root): python -I extensions/scripts/occupancy_sensitivity.py data/raw/room_occupancy_estimation/derived_60s/occupancy_60s.csv OUT.json 1000 200 (and 50)
"""
import csv, json, sys
from collections import defaultdict
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from c11_upgrade.population import X, Y  # noqa: E402
from c11_upgrade.addendum_core import population_references, solve_information  # noqa: E402
from c11_upgrade.inference import cp_upper  # noqa: E402

STRATA = {'weekday 09-17': (True, True), 'all days 09-17': (False, True),
          'weekday all hours': (True, False), 'all days all hours': (False, False)}
LAGS = (1, 2, 3, 5)
ENCODE = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(X, Y))}


def load(path):
    days = defaultdict(dict)
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r['timestamp'])
            x = int(int(r['count']) > 0); y = int(int(r['S6_PIR']) > 0 or int(r['S7_PIR']) > 0)
            days[t.date()][t] = (x, y)
    return days


def windows(days, weekday_only, office_hours, lag):
    d = timedelta(minutes=lag); out = []
    for day, pts in sorted(days.items()):
        for c in sorted(pts):
            if weekday_only and c.weekday() >= 5: continue
            ts = (c - d, c, c + d)
            if office_hours and not all(9 <= t.hour < 17 for t in ts): continue
            if all(t in pts and t.date() == day for t in ts):
                out.append((day, [pts[t] for t in ts]))
    return out


def finite_sample(ws, theta, reps, budget):
    yy = np.asarray([[p[1] for p in w] for _, w in ws], int); xx = np.asarray([w[1][0] for _, w in ws], int)
    n = len(ws); acc = {k: dict(contains=0, width=0.0, empty=0, failure=0) for k in 'MNF'}
    for r in range(reps):
        order = np.argsort(np.random.default_rng(470001 + r).random(n), kind='stable')
        aud, cal = order[:budget], order[budget:2 * budget]
        eps = cp_upper(int(np.sum((xx[cal] == 0) & (yy[cal, 1] == 1))), budget, .01)
        for k, iv in solve_information(yy, xx[aud], yy[aud], eps, .04).items():
            # An empty set (infeasibility certified by the solver) is reported as the flagged interval [-1, 1]; any other
            # unsuccessful status is a solver failure: counted separately, width 2, and never counted as containing the target.
            status = iv['status']
            if status not in ('ok', 'empty', 'solver_failure'): raise ValueError(f'unrecognized solver status {status!r}')
            lo, hi = (iv['lower'], iv['upper']) if status == 'ok' else (-1.0, 1.0)
            acc[k]['empty'] += status == 'empty'; acc[k]['failure'] += status == 'solver_failure'
            acc[k]['contains'] += status != 'solver_failure' and bool(lo - 1e-9 <= theta <= hi + 1e-9); acc[k]['width'] += hi - lo
    return {k: dict(contains_record_target=v['contains'] / reps, mean_width=v['width'] / reps, empty_sets=v['empty'], solver_failures=v['failure'])
            for k, v in acc.items()}


def summarize(ws):
    n = len(ws)
    if n == 0: return dict(n=0)
    counts = np.zeros((4, 4, 4), int)
    for _, w in ws: counts[tuple(ENCODE[p] for p in w)] += 1
    law = np.array([Fraction(int(v), n) for v in counts.ravel()], dtype=object).reshape(4, 4, 4)
    ref = population_references(law)
    changes = sum(w[0][0] != w[1][0] for _, w in ws)
    occ = [w[1] for _, w in ws if w[1][0] == 1]
    out = dict(n=n, days=len({d for d, _ in ws}), theta_record=ref['theta'], changes_first_pair=int(changes),
               f_hat=ref['f'], stationarity_residual=ref['stationarity_residual'],
               occupied_share=float(np.mean([w[1][0] for _, w in ws])),
               false_negative_rate=float(np.mean([y == 0 for _, y in occ])) if occ else None, intervals={})
    for info, iv in ref['intervals'].items():
        e = dict(status=iv['status'])
        if iv['status'] == 'ok':
            e.update(lower=iv['lower'], upper=iv['upper'], width=iv['upper'] - iv['lower'],
                     min_changes_at_upper=(1 - iv['upper']) * n / 2,
                     record_inside=bool(iv['lower'] - 1e-12 <= ref['theta'] <= iv['upper'] + 1e-12),
                     record_gap=max(0.0, ref['theta'] - iv['upper'], iv['lower'] - ref['theta']))
        out['intervals'][info] = e
    return out


if __name__ == '__main__':
    days = load(sys.argv[1]); res = []
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 0; budget = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    for name, (wk, oh) in STRATA.items():
        for lag in LAGS:
            ws = windows(days, wk, oh, lag); row = dict(stratum=name, lag_minutes=lag, **summarize(ws))
            if reps and len(ws) >= 2 * budget:
                row['finite_sample'] = dict(replicates=reps, budget=budget, **finite_sample(ws, row['theta_record'], reps, budget))
            res.append(row); print(name, lag, 'done', flush=True, file=sys.stderr)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1, default=float))
    for r in res:
        iv = r.get('intervals', {})
        f = lambda k: (f"[{iv[k]['lower']:.4f},{iv[k]['upper']:.4f}]" if iv.get(k, {}).get('status') == 'ok' else iv.get(k, {}).get('status', '-'))
        print(f"{r['stratum']:20s} lag{r['lag_minutes']} n={r['n']:5d} days={r.get('days')} theta={r.get('theta_record', float('nan')):.4f} "
              f"chg={r.get('changes_first_pair')} f={r.get('f_hat', float('nan')):.4f} res={r.get('stationarity_residual', float('nan')):.4f} "
              f"M={f('M')} N={f('N')} F={f('F')}")
        fs = r.get('finite_sample')
        if fs: print('   finite B=%d x%d: ' % (fs['budget'], fs['replicates']) + '  '.join(
            f"{k}: contains {fs[k]['contains_record_target']:.3f} width {fs[k]['mean_width']:.3f} empty {fs[k]['empty_sets']} failures {fs[k]['solver_failures']}" for k in 'MNF'))
