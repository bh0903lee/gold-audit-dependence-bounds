"""Six-variable comparison: six-variable program vs neighbor word program and external comparators.

Implements the prespecified analysis; written and smoke-tested before the main run. Datasets are never pooled:
'confirm' (verdict on the primary and secondary criteria, keys P1/P2), 'extended' (confirm plus extension seeds, for
coverage precision) and, when given, 'reanalysis_v42' (main-study arrays; a reanalysis, not a confirmation).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as w

REF = 'word/exact_binomial'; CAND = 'sixvar/exact_binomial'; PRIMARY_BUDGET = 500
PAIRS = [(REF, CAND, 'P1/P2: sixvar CP vs word CP'), ('word/hoeffding', 'sixvar/hoeffding', 'sixvar vs word, Hoeffding'),
         ('word/empirical_bernstein', 'sixvar/empirical_bernstein', 'sixvar vs word, EB'), ('sixvar/hoeffding', CAND, 'CP vs Hoeffding inside sixvar'),
         (REF, 'l1ball', 'L1-ball vs word CP'), (REF, 'bootstrap', 'bootstrap vs word CP'), (REF, 'dirichlet', 'Dirichlet vs word CP'),
         (CAND, 'bootstrap', 'bootstrap vs sixvar CP'), (CAND, 'dirichlet', 'Dirichlet vs sixvar CP')]


def analyse(rows):
    cells = defaultdict(list)
    for q in rows: cells[(q['stay'], q['audits'], q['method'])].append(q)
    stays = sorted({k[0] for k in cells}); budgets = sorted({k[1] for k in cells}); methods = sorted({k[2] for k in cells})
    table = []; pairs = []
    for st in stays:
        for B in budgets:
            for mth in methods:
                rr = cells[(st, B, mth)]; c = w.cell_summary(rr)
                contain = sum(bool(q['contains_population']) for q in rr)
                c.update(stay=st, audits=B, method=mth, guarantee=rr[0]['guarantee'], contains_population=contain/len(rr), wilson_95_contains=w.wilson(contain, len(rr)),
                         empty_intervals=sum(bool(q.get('empty_interval')) for q in rr), fallbacks=sum(bool(q.get('fallback')) for q in rr),
                         mean_discard_fraction=(sum(q['discard_fraction'] for q in rr)/len(rr) if 'discard_fraction' in rr[0] else None),
                         population_width=rr[0]['population_high'] - rr[0]['population_low'])
                table.append(c)
            for ref, cand, label in PAIRS:
                if (st, B, ref) in cells and (st, B, cand) in cells:
                    pairs.append(dict(stay=st, audits=B, reference=ref, candidate=cand, label=label, **w.paired_reduction(cells[(st, B, ref)], cells[(st, B, cand)])))
    return dict(seeds=len({q['seed'] for q in rows}), seed_range=[min(q['seed'] for q in rows), max(q['seed'] for q in rows)], cells=table, paired_reductions=pairs)


def verdict(a):
    """Prespecified criteria at B=500 from one dataset."""
    cell = {(c['stay'], c['audits'], c['method']): c for c in a['cells']}; per = []
    for p in a['paired_reductions']:
        if p['audits'] != PRIMARY_BUDGET or (p['reference'], p['candidate']) != (REF, CAND): continue
        c = cell[(p['stay'], PRIMARY_BUDGET, CAND)]; lo, hi = p['interval']
        per.append(dict(stay=p['stay'], mean_reduction_pp=p['mean'], interval=p['interval'], coverage=c['coverage'], contains_population=c['contains_population'],
                        reduction_ge_3=p['mean'] >= 3, lower_gt_0=lo > 0, coverage_ok=c['coverage'] >= .95, containment_ok=c['contains_population'] >= .95, reduction_ge_10=p['mean'] >= 10,
                        negative=('wider' if (p['mean'] < 0 and hi < 0) else 'interval_contains_0' if lo <= 0 <= hi else None)))
    P1 = bool(per) and all(x['reduction_ge_3'] and x['lower_gt_0'] and x['coverage_ok'] and x['containment_ok'] for x in per)
    P2 = sum(x['reduction_ge_10'] for x in per) >= 2
    return dict(budget=PRIMARY_BUDGET, per_stay=per, P1=P1, P2=P2, negative_regimes=[x['stay'] for x in per if x['negative']],
                wording=('materially shorter' if P1 and P2 else 'uniformly, modestly shorter' if P1 else 'P1 not met'))


def comparator_validity(a, budgets=(100, 500)):
    out = []
    for c in a['cells']:
        if c['method'] in ('bootstrap', 'dirichlet') and c['audits'] in budgets:
            ok = c['coverage'] >= .95 and c['wilson_95'][0] >= .93
            out.append(dict(stay=c['stay'], audits=c['audits'], method=c['method'], coverage=c['coverage'], wilson_95=c['wilson_95'], n=c['n'], valid_for_width_comparison=bool(ok),
                            label=None if ok else 'shorter but invalid'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); p.add_argument('--plan', type=Path)
    p.add_argument('--confirm-range', type=int, nargs=2, default=[92001, 92200]); p.add_argument('--reanalysis', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('Refusing to overwrite')
    rows, info = w.load_run(a.run, a.plan)
    if any(q['experiment'] != 'w3' for q in rows): p.error('Not a six-variable comparison run')
    lo, hi = a.confirm_range; confirm = [q for q in rows if lo <= q['seed'] <= hi]
    result = dict(protocol='prespecified analysis (2026-10-13)', run=info, confirm=analyse(confirm), extended=analyse(rows))
    result['confirm']['verdict'] = verdict(result['confirm']); result['extended']['verdict_exploratory'] = verdict(result['extended'])
    result['extended']['comparator_validity'] = comparator_validity(result['extended'])
    if a.reanalysis:
        old = json.loads(a.reanalysis.read_text()); result['reanalysis_v42'] = analyse(old); result['reanalysis_v42']['verdict'] = verdict(result['reanalysis_v42'])
        result['reanalysis_v42']['note'] = 'Main-study arrays (seeds 91001-91200); the word-program rows had already been reported, so this is a reanalysis, not a confirmation.'
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    for name in ('confirm', 'reanalysis_v42'):
        if name in result:
            v = result[name]['verdict']; print(name, 'P1', v['P1'], 'P2', v['P2'], [(x['stay'], round(x['mean_reduction_pp'], 2), [round(t, 2) for t in x['interval']], x['coverage'], x['contains_population']) for x in v['per_stay']])


if __name__ == '__main__':
    main()
