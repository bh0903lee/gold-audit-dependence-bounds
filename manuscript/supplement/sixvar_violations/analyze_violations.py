"""Assumption-violation sensitivity study: population breakdown map, finite-sample coverage, refuting-audit power.

Design in docs/STUDY_DESIGN.md. The 1000-seed set (w2a) and the 200-seed set (w2b) are never pooled. The breakdown map and the
fine-grid curves are exact population computations and use no simulated data.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as w
from topic_experiment import violations as v

STAYS = (.2, .5, .8); MAIN = 'word/exact_binomial'; SIX = 'sixvar/exact_binomial'; TOL = 1e-9
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


def paired_widths(by):
    out = []
    for (axis, level, stay, B, method), rr in by.items():
        if method == MAIN and (axis, level, stay, B, SIX) in by:
            out.append(dict(axis=axis, level=level, stay=stay, audits=B, label='sixvar CP vs word CP', **w.paired_reduction(rr, by[(axis, level, stay, B, SIX)])))
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
    result = dict(breakdown_map=bmap, breakdown_curves=breakdown_curves(),
                  set_a=dict(run=ia, seeds=len({q['seed'] for q in ra}), cells=ca, paired_width=paired_widths(bya)),
                  set_b=dict(run=ib, seeds=len({q['seed'] for q in rb}), cells=cb, paired_width=paired_widths(byb)))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps(dict(seeds_a=result['set_a']['seeds'], seeds_b=result['set_b']['seeds'], cells_a=len(ca), cells_b=len(cb))))


if __name__ == '__main__':
    main()
