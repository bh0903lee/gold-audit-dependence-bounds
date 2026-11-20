"""Six-variable comparison: six-variable program vs neighbor word program and external comparators.

Design in docs/STUDY_DESIGN.md. Datasets are never pooled: 'width_comparison' (the 200 width-comparison seeds),
'extended' (these plus the further seeds, for coverage precision) and, when given, 'reanalysis_v42' (the six-variable
methods applied to the main-study arrays).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as w

REF = 'word/exact_binomial'; CAND = 'sixvar/exact_binomial'; PRIMARY_BUDGET = 500
PAIRS = [(REF, CAND, 'sixvar CP vs word CP'), ('word/hoeffding', 'sixvar/hoeffding', 'sixvar vs word, Hoeffding'),
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


def comparator_coverage(a, budgets=(100, 500)):
    """Coverage of the two constructions without a guarantee, with Wilson intervals."""
    return [dict(stay=c['stay'], audits=c['audits'], method=c['method'], coverage=c['coverage'], wilson_95=c['wilson_95'], n=c['n'])
            for c in a['cells'] if c['method'] in ('bootstrap', 'dirichlet') and c['audits'] in budgets]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); p.add_argument('--plan', type=Path)
    p.add_argument('--width-range', type=int, nargs=2, default=[92001, 92200]); p.add_argument('--reanalysis', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('Refusing to overwrite')
    rows, info = w.load_run(a.run, a.plan)
    if any(q['experiment'] != 'w3' for q in rows): p.error('Not a six-variable comparison run')
    lo, hi = a.width_range; width = [q for q in rows if lo <= q['seed'] <= hi]
    expected = set(range(lo, hi + 1)); cells = defaultdict(set)  # the width comparison needs every one of its seeds in every cell
    for q in width: cells[(q['stay'], q['audits'], q['method'])].add(q['seed'])
    keys = {(q['stay'], q['audits'], q['method']) for q in rows}
    if not width or set(cells) != keys or any(s != expected for s in cells.values()):
        p.error(f'Incomplete width-comparison seeds {lo}-{hi}: every (stay, audits, method) cell needs all of them; nothing written')
    result = dict(run=info, width_comparison=analyse(width), extended=analyse(rows))
    result['extended']['comparator_coverage'] = comparator_coverage(result['extended'])
    if a.reanalysis:
        old = json.loads(a.reanalysis.read_text()); result['reanalysis_v42'] = analyse(old)
        result['reanalysis_v42']['note'] = 'Main-study arrays (seeds 91001-91200), analysed separately from the width-comparison seeds.'
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    for name in ('width_comparison', 'reanalysis_v42'):
        if name in result:
            print(name, [(x['stay'], round(x['mean'], 2), [round(t, 2) for t in x['interval']]) for x in result[name]['paired_reductions']
                         if x['audits'] == PRIMARY_BUDGET and (x['reference'], x['candidate']) == (REF, CAND)])


if __name__ == '__main__':
    main()
