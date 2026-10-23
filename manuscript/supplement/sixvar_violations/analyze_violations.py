"""Assumption-violation sensitivity study: population breakdown map, finite-sample coverage, refuting-audit power, criteria R1-R6.

Implements the prespecified analysis; written and smoke-tested before the main
run. The 1000-seed set (w2a) and the 200-seed set (w2b) are never pooled. The breakdown map and the
fine-grid curves are exact population computations and use no simulated data.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as w
from topic_experiment import violations as v

STAYS = (.2, .5, .8); MAIN = 'word/exact_binomial'; SIX = 'sixvar/exact_binomial'; KEY_BUDGET = 500; TOL = 1e-9
CURVES = {'V1': ('V1', np.linspace(0, .10, 201), float), 'V2a_stay': ('V2a', np.linspace(0, .20, 201), lambda x: f'stay+{x!r}'),
          'V2a_det': ('V2a', np.linspace(0, .10, 201), lambda x: f'det+{x!r}'), 'V2b': ('V2b', np.linspace(0, .20, 201), float),
          'V4a': ('V4a', np.linspace(0, .05, 201), float), 'V4b': ('V4b', np.linspace(0, .20, 201), float), 'V5': ('V5', np.linspace(1, 4, 301), float)}


def breakdown_map():
    out = []
    for stay in STAYS:
        for axis, level in v.grid(list(v.LEVELS)):
            out.append(dict(axis=axis, level=str(level), stay=stay, in_model=axis in v.IN_MODEL, **v.breakdown(v.make_condition(axis, level, stay))))
    return out


def breakdown_curves():
    """inf{level: d>0} and the first level with incompatible contaminated moments, on a fine grid (grade-free)."""
    out = []
    for name, (axis, grid, enc) in CURVES.items():
        for stay in STAYS:
            ds = []; comp = []
            for x in grid:
                b = v.breakdown(v.make_condition(axis, enc(float(x)), stay), word_lp=False); ds.append(b['d']); comp.append(b['compatible'])
            ds = np.array(ds); pos = np.flatnonzero(ds > TOL); bad = np.flatnonzero(~np.array(comp))
            out.append(dict(curve=name, stay=stay, grid=[float(grid[0]), float(grid[-1]), len(grid)], first_level_with_d_positive=None if not len(pos) else float(grid[pos[0]]),
                            last_level_with_d_zero=None if not len(pos) else float(grid[max(pos[0] - 1, 0)]), d_at_grid_end=float(ds[-1]), max_d=float(ds.max()),
                            first_incompatible_level=None if not len(bad) else float(grid[bad[0]]), d=[float(x) for x in ds[::10]], levels=[float(x) for x in grid[::10]]))
    return out


def cells(rows, exact):
    by = defaultdict(list)
    for q in rows: by[(q['axis'], q['level'], q['stay'], q['audits'], q['method'])].append(q)
    out = []
    for (axis, level, stay, B, method), rr in sorted(by.items(), key=lambda kv: tuple(map(str, kv[0]))):
        c = w.cell_summary(rr); ex = exact[(axis, level, stay)]; rej = sum(q['t1_count'] > 0 for q in rr); passed = [q for q in rr if q['t1_count'] == 0]
        c.update(axis=axis, level=level, stay=stay, audits=B, method=method, d=ex['d'], compatible=ex['compatible'], word_feasible=ex['word_feasible'],
                 t1_rejections=rej, t1_rejection_rate=rej/len(rr), t1_power_analytic=1 - (1 - ex['t1_probability'])**B,
                 t1_pass_n=len(passed), t1_pass_coverage=(sum(bool(q['covered']) for q in passed)/len(passed) if passed else None),
                 t1_pass_wilson_95=w.wilson(sum(bool(q['covered']) for q in passed), len(passed)))
        out.append(c)
    return out, by


def criteria(ca, cb, exact):
    A = {(c['axis'], c['level'], c['stay'], c['audits'], c['method']): c for c in ca}; Bc = {(c['axis'], c['level'], c['stay'], c['audits'], c['method']): c for c in cb}
    def cov(table, axis, level, stay, budget=KEY_BUDGET, method=MAIN): return table[(axis, str(level), stay, budget, method)]
    res = {}
    # R1: in-model axes (200-seed set).
    r1 = [dict(axis=a, level=str(l), stay=s, d=exact[(a, str(l), s)]['d'], coverage=cov(Bc, a, l, s)['coverage'], wilson_low=cov(Bc, a, l, s)['wilson_95'][0],
               coverage_sixvar=cov(Bc, a, l, s, method=SIX)['coverage']) for a in ('V3a', 'V3b') for l in v.LEVELS[a] for s in STAYS]
    res['R1'] = dict(passed=all(x['d'] <= TOL and x['wilson_low'] >= .92 for x in r1), cells=r1)
    # R2: false positives (V1).
    r2 = []
    for s in STAYS:
        row = [(l, cov(A, 'V1', l, s)) for l in v.LEVELS['V1']]; keep90 = None; keep95 = None
        for l, c in row:                       # largest grade such that every grade up to it keeps the criterion
            if c['coverage'] >= .90: keep90 = l
            else: break
        for l, c in row:
            if c['t1_pass_coverage'] is not None and c['t1_pass_coverage'] >= .95: keep95 = l
            else: break
        r2.append(dict(stay=s, max_phi_coverage_ge_090=keep90, max_phi_t1_pass_coverage_ge_095=keep95, coverage_at_min_grade=row[0][1]['coverage'],
                       by_level=[dict(phi=l, coverage=c['coverage'], wilson_95=c['wilson_95'], t1_rejection_rate=c['t1_rejection_rate'], t1_pass_n=c['t1_pass_n'], t1_pass_coverage=c['t1_pass_coverage'], d=c['d']) for l, c in row]))
    res['R2'] = dict(per_stay=r2, negative_extremely_sensitive=any(x['coverage_at_min_grade'] < .90 for x in r2))
    # R3: audit-cohort drift (V2a).
    def lv(l): return [dict(stay=s, coverage=cov(A, 'V2a', l, s)['coverage'], wilson_95=cov(A, 'V2a', l, s)['wilson_95'], d=exact[('V2a', l, s)]['d']) for s in STAYS]
    res['R3'] = dict(small_drift_tolerated=all(x['coverage'] >= .90 for x in lv('stay+0.05')), concurrency_required=any(x['coverage'] < .90 for x in lv('stay+0.1')),
                     by_level={l: lv(l) for l in v.LEVELS['V2a']})
    # R4: audit label flips and anchoring (V4a, V4b).
    res['R4'] = dict(v4a_eps_001_coverage_ge_090=all(cov(A, 'V4a', .01, s)['coverage'] >= .90 for s in STAYS),
                     v4a_eps_001=[dict(stay=s, coverage=cov(A, 'V4a', .01, s)['coverage'], wilson_95=cov(A, 'V4a', .01, s)['wilson_95']) for s in STAYS],
                     d_at_eps_005=[dict(stay=s, v4a=exact[('V4a', '0.05', s)]['d'], v4b=exact[('V4b', '0.05', s)]['d'], anchoring_larger=exact[('V4b', '0.05', s)]['d'] > exact[('V4a', '0.05', s)]['d']) for s in STAYS])
    # R5: outcome-dependent selection (V5).
    res['R5'] = dict(per_stay=[dict(stay=s, d_rho2=exact[('V5', '2', s)]['d'], d_positive=exact[('V5', '2', s)]['d'] > TOL, compatible=exact[('V5', '2', s)]['compatible'],
                                    coverage_rho2=cov(A, 'V5', 2, s)['coverage'], coverage_rho2_empty_as_miss=cov(A, 'V5', 2, s)['coverage_empty_as_miss']) for s in STAYS])
    # R6: windows from one sequence (V6; 200-seed set), step 1.
    r6 = [dict(stay=s, audits=b, coverage=cov(Bc, 'V6', 1, s, b)['coverage'], wilson_95=cov(Bc, 'V6', 1, s, b)['wilson_95']) for s in STAYS for b in sorted({k[3] for k in Bc})]
    res['R6'] = dict(some_budget_below_095=any(x['coverage'] < .95 for x in r6), cells=r6)
    return res


def paired_widths(by):
    out = []
    for (axis, level, stay, B, method), rr in by.items():
        if method == MAIN and (axis, level, stay, B, SIX) in by:
            out.append(dict(axis=axis, level=level, stay=stay, audits=B, label='sixvar CP vs word CP (exploratory)', **w.paired_reduction(rr, by[(axis, level, stay, B, SIX)])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('a', 'b'): p.add_argument(f'--run-{name}', type=Path, required=True); p.add_argument(f'--plan-{name}', type=Path)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    if a.out.exists(): p.error('Refusing to overwrite')
    bmap = breakdown_map(); exact = {(b['axis'], b['level'], b['stay']): b for b in bmap}
    ra, ia = w.load_run(a.run_a, a.plan_a); rb, ib = w.load_run(a.run_b, a.plan_b)
    if {q['experiment'] for q in ra + rb} != {'w2'} or {q['seed'] for q in ra} & {q['seed'] for q in rb}: p.error('Expected two disjoint assumption-violation runs')
    ca, bya = cells(ra, exact); cb, byb = cells(rb, exact)
    result = dict(protocol='prespecified analysis (2026-10-13)', breakdown_map=bmap, breakdown_curves=breakdown_curves(),
                  set_a=dict(run=ia, seeds=len({q['seed'] for q in ra}), cells=ca, paired_width=paired_widths(bya)),
                  set_b=dict(run=ib, seeds=len({q['seed'] for q in rb}), cells=cb, paired_width=paired_widths(byb)), criteria=criteria(ca, cb, exact))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    c = result['criteria']; print(json.dumps(dict(R1=c['R1']['passed'], R2=[(x['stay'], x['max_phi_coverage_ge_090'], x['max_phi_t1_pass_coverage_ge_095']) for x in c['R2']['per_stay']],
          R2_negative=c['R2']['negative_extremely_sensitive'], R3=[c['R3']['small_drift_tolerated'], c['R3']['concurrency_required']], R4=c['R4']['v4a_eps_001_coverage_ge_090'],
          R5=[x['d_positive'] for x in c['R5']['per_stay']], R6=c['R6']['some_budget_below_095'])))


if __name__ == '__main__':
    main()
