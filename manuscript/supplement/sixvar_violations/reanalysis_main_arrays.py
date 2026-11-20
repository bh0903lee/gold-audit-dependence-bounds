"""Apply the six-variable comparison methods to the main-study L=3 arrays and check the word-program regression.

Reads a main-study run archive (never writes there). For every array the neighbor word-program rows are recomputed
with the current engine and compared exactly with the archived rows; the six-variable programs and comparators are
computed on the same arrays. Output goes under results/generated/.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as w
from topic_experiment import sixvar, violations, w3w2, runtime

ROOT = w.ROOT
OLD = {'word/hoeffding': 'neighbor/hoeffding', 'word/empirical_bernstein': 'neighbor/empirical_bernstein', 'word/exact_binomial': 'neighbor/exact_binomial'}


def one_run(run):
    run = Path(run); manifest = runtime.verify_outputs(run)                      # PASS + intact hashes (arrays included)
    methods = json.loads((ROOT/'configs/main_w3.json').read_text())['methods']
    old = {(q['seed'], q['stay'], q['audits'], q['method']): q for q in json.loads((run/'rows.json').read_text()) if q['L'] == 3 and q['source'] == 'both_sampled'}
    rows = []; mismatches = []; compared = 0
    for f in sorted(run.glob('*_3_*.npz')):
        seed, stay, _, budget = f.stem.split('_'); seed = int(seed); stay = float(stay); budget = int(budget); z = np.load(f)
        law = violations.analyst_law(violations.make_condition('baseline', 0, stay)); pop = [float(v) for v in sixvar.closed_form(*[law[k] for k in sixvar.NAMES])]
        for method in methods:
            new = w3w2.evaluate(method, z['y'], z['audit_x'], z['audit_y'], .05, [seed, int(round(1000*stay)), budget], 2*stay - 1, pop)
            if method in OLD:
                o = old[(seed, stay, budget, OLD[method])]; compared += 1
                if (new['low'], new['high'], new['empty_confidence_set'], new['covered']) != (o['low'], o['high'], o['empty_confidence_set'], o['covered']):
                    mismatches.append(dict(file=str(f.relative_to(ROOT)), method=method, new=[new['low'], new['high']], archived=[o['low'], o['high']]))
            rows.append(dict(experiment='w3', dataset='reanalysis_v42', seed=seed, stay=stay, L=3, audits=budget, method=method, guarantee=w3w2.GUARANTEE[method.split('/')[0]],
                             truth=2*stay - 1, population_low=pop[0], population_high=pop[1], status='OK', **new))
    return rows, mismatches, compared, manifest['config']['seeds']


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=24); p.add_argument('--limit', type=int); p.add_argument('--runs', type=Path, default=ROOT/'results/runs/v4_2/main')
    a = p.parse_args(); out = a.out.resolve()
    if not 1 <= a.workers <= 24: p.error('workers must be in 1..24')
    if not out.is_relative_to((ROOT/'results/generated').resolve()) or out.exists(): p.error('--out must be a new directory under results/generated')
    runs = sorted(q.parent for q in a.runs.glob('*/C11-*/attempt_*/manifest.json') if json.loads(q.read_text())['status'] == 'PASS')[:a.limit]
    with Pool(a.workers) as pool: parts = pool.map(one_run, [str(r) for r in runs], chunksize=1)
    rows = [q for part in parts for q in part[0]]; mismatches = [q for part in parts for q in part[1]]; seeds = sorted(s for part in parts for s in part[3])
    out.mkdir(parents=True); (out/'rows.json').write_text(json.dumps(rows, allow_nan=False) + '\n')
    report = dict(status='PASS' if not mismatches and len(seeds) == len(set(seeds)) else 'FAIL', archived_runs=len(runs), seeds=len(seeds), seed_range=[seeds[0], seeds[-1]],
                  word_rows_compared_exactly=sum(part[2] for part in parts), mismatches=mismatches, rows=len(rows), rows_sha256=runtime.sha(out/'rows.json'))
    (out/'regression.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps({k: v for k, v in report.items() if k != 'mismatches'} | {'mismatches': len(mismatches)}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    sys.exit(main())
