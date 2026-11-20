"""Data preparation for the main simulation study: collect the per-seed outputs of a run archive into the stored row,
slack-attribution and input-manifest files. Requires the simulation arrays (run scripts/execute_plan.py first).

The finite-sample evidence is the main simulation collection (200 seed jobs, 90,000 rows; empty confidence sets
recorded as flagged [-1,1] fallbacks). The slack attribution recomputes the Clopper-Pearson bounds of the neighbor
audit at L=3, B=500 from the archived arrays and checks them against the stored rows.

  python scripts/freeze_main_rows.py --runs results/generated/main --out DIR
writes DIR/main_study_rows.csv, DIR/main_study_slack_attribution.json and DIR/main_study_input_manifest.json; the first
two equal the files in manuscript/supplement/ for a re-run with the same configuration and seeds (the manifest records
the paths and SHA-256 digests of the run archive used). Then `python scripts/recompute_values.py --supplement` on a copy
of manuscript/supplement/ holding these files recomputes the reported values.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / 'manuscript/supplement'
FREEZE_DATE = '2026-10-13'
FIELDS = ['stage', 'seed', 'stay', 'L', 'audits', 'moments', 'interval', 'source', 'status', 'empty_confidence_set', 'error', 'low', 'high', 'width', 'covered',
          'population_low', 'population_high', 'population_width', 'concentration_slack']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, default=ROOT / 'results/generated/main', help='run archive produced by scripts/execute_plan.py')
    p.add_argument('--out', type=Path, required=True, help='new output directory')
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from topic_experiment import engine as e  # simulation engine (read-only import)
    inputs = []; rows = []; slack = []
    files = sorted(args.runs.glob('*/C11-*/attempt_01/rows.json'))
    assert len(files) == 200, ('v4_2', len(files))
    for f in files:
        manifest = json.loads((f.parent / 'manifest.json').read_text()); assert manifest['status'] == 'PASS', f
        assert manifest['output_hashes']['rows.json'] == sha(f), f
        raw = json.loads(f.read_text()); assert len(raw) == 450, (f, len(raw))
        inputs.append({'path': rel(f), 'sha256': sha(f)})
        for r in raw:
            rows.append(dict(stage='v42', seed=r['seed'], stay=r['stay'], L=r['L'], audits=r['audits'], moments=r['moments'], interval=r['interval'], source=r['source'], status=r['status'],
                             empty_confidence_set=r['empty_confidence_set'], error=r.get('error', ''), low=r['low'], high=r['high'], width=r['width'], covered=r['covered'],
                             population_low=r['population_low'], population_high=r['population_high'], population_width=r['population_width'], concentration_slack=r['concentration_slack']))
        # Fixed-allocation two-group Shapley attribution of CP slack at L=3, B=500 (neighbor), from the archived arrays.
        seed = raw[0]['seed']; g = {(r['stay'], r['L'], r['audits'], r['method']): r for r in raw}
        for stay in [.2, .5, .8]:
            npz = f.parent / f'{seed}_{stay}_3_500.npz'; assert manifest['output_hashes'][npz.name] == sha(npz), npz
            inputs.append({'path': rel(npz), 'sha256': sha(npz)})
            z = np.load(npz); y, ax, ay = z['y'], z['audit_x'], z['audit_y']
            cp = g[stay, 3, 500, 'neighbor/exact_binomial']; pop = cp['population_width']
            clo, chi = e.bounds(y, ax, ay, interval='exact_binomial'); assert abs(clo - cp['low']) < 1e-12 and abs(chi - cp['high']) < 1e-12, (seed, stay)
            obs, aud = e.population_moments(stay, [.95, .2], 3, 'neighbor')
            # Mtotal=11 (8 observation patterns + 3 audit features): every sampled feature keeps delta=.05/11.
            eo = e.bounds(y, ax, ay, alpha=.05 * 3 / 11, interval='exact_binomial', exact_obs=obs)
            ea = e.bounds(y, ax, ay, alpha=.05 * 8 / 11, interval='exact_binomial', exact_audit=aud)
            wa = eo[1] - eo[0]; wo = ea[1] - ea[0]; wb = cp['width']; total = wb - pop
            slack.append(dict(seed=seed, stay=stay, interval='exact_binomial', L=3, audits=500, width_both=wb, width_audit_sampled_only=wa, width_obs_sampled_only=wo, population_width=pop,
                              total_slack=total, audit_shapley=(wa - pop + wb - wo) / 2, observation_shapley=(wo - pop + wb - wa) / 2, interaction=wb - wa - wo + pop, audit_fraction=(wa - pop + wb - wo) / (2 * total)))
    assert len(rows) == 90000 and len(slack) == 600
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'main_study_rows.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
    (args.out / 'main_study_slack_attribution.json').write_text(json.dumps({'definition': 'Two-group Shapley attribution of CP excess width with fixed per-feature error delta=.05/11; computed from the archived v4.2 arrays at L=3, B=500, neighbor audit', 'rows': slack}, indent=2) + '\n')
    for path in (SUP / 'main_study_independent_replay.json', SUP / 'verify_theorem.json'):
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    (args.out / 'main_study_input_manifest.json').write_text(json.dumps({'freeze_date': FREEZE_DATE, 'collection': 'results/runs/v4_2 (200 seed jobs, 90,000 rows)', 'inputs': inputs}, indent=2) + '\n')
    print(f'wrote {len(rows)} rows and {len(slack)} slack attributions to {args.out}')


if __name__ == '__main__':
    main()
