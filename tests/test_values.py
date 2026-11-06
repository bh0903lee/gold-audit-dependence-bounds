"""Tests for the value check (scripts/recompute_values.py): the clean package passes; a deleted, extra or changed stored file,
a changed reported number and a corrupted value file each fail; inventory expressions are restricted."""
import importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; SUP = ROOT / 'manuscript/supplement'; DATA = ROOT / 'data/raw'; SCRIPT = ROOT / 'scripts/recompute_values.py'
spec = importlib.util.spec_from_file_location('recompute_values', SCRIPT); rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)


def run(supplement):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--supplement', str(supplement)], capture_output=True, text=True, env=env)


class Copy:
    """A scratch copy of manuscript/supplement/ that a test may corrupt."""
    def __enter__(self):
        self.td = tempfile.TemporaryDirectory(); self.sup = Path(self.td.name) / 'supplement'
        shutil.copytree(SUP, self.sup, ignore=shutil.ignore_patterns('__pycache__')); return self.sup

    def __exit__(self, *exc): self.td.cleanup()


def edit_json(path, change):
    value = json.loads(path.read_text()); change(value); path.write_text(json.dumps(value, indent=1))


def rehash(sup, name):
    """Record the current digest of a changed file, so that only the semantic checks can catch the change."""
    edit_json(sup / 'supplement_manifest.json', lambda m: [r.update(sha256=rv.sha256(sup / name)) for r in m['files'] if r['path'] == name])


class Values(unittest.TestCase):
    def test_extension_and_sixvar_values_equal_stored_files(self):
        for part in (rv.sixvar_violations, rv.extensions):
            for name, text in part(SUP).items():
                self.assertEqual(text, (SUP / name).read_bytes().decode(), name)

    def test_clean_package_passes_integrity_and_inventory(self):
        self.assertEqual(list(rv.integrity(SUP, DATA)), [])
        self.assertEqual(list(rv.reported(SUP)), [])

    def test_inventory_reads_every_summary_quoted_in_the_article(self):
        sources = ' '.join(e['source'] for e in json.loads((SUP / 'reported_values.json').read_text())['entries'])
        for name in ('sphere_summary.json', 'paired_budget_summary.json', 'application_sensitivity_B50.json', 'application_sensitivity_B200.json',
                     'fp_population_rows.jsonl', 'laws_summary.json', 'overlap_summary.json', 'fp_budget_summary.json', 'application_summary.json'):
            self.assertIn(name, sources)

    def test_tolerance_and_structure(self):
        self.assertEqual(list(rv.differences({'a': [1.0, 'x']}, {'a': [1.0 + 1e-12, 'x']})), [])
        self.assertEqual(list(rv.differences({'a': [1.0]}, {'a': [1.001]})), ['$.a[0]'])
        self.assertTrue(list(rv.differences({'a': 1}, {'b': 1})))
        self.assertTrue(list(rv.differences([True], [1])))

    def test_printed_value_rules(self):
        self.assertTrue(rv.agrees(0.17549, '0.18')); self.assertFalse(rv.agrees(0.17549, '0.17'))
        self.assertTrue(rv.agrees(-0.1, '-1/10')); self.assertFalse(rv.agrees(-0.1, '1/10'))
        self.assertTrue(rv.agrees(1.4e-15, '1.56e-15', 'below')); self.assertFalse(rv.agrees(0.998, '0.999', 'at_least'))
        self.assertFalse(rv.agrees(float('nan'), '0.5')); self.assertFalse(rv.agrees(True, '1'))

    def test_expressions_are_restricted(self):
        evaluate = rv.evaluator(SUP)
        self.assertEqual(evaluate("len(J('extensions/application_sensitivity_B50.json'))"), 16)
        for bad in ("J('main_study_summary.json').keys()", "open('x')", "().__class__", "(lambda: 1)()"):
            with self.assertRaises(ValueError, msg=bad): evaluate(bad)

    def test_deleted_file_fails(self):
        with Copy() as sup:
            (sup / 'extensions/sphere_summary.json').unlink()
            bad = run(sup)
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
        self.assertIn('missing file   manuscript/supplement/extensions/sphere_summary.json', bad.stdout)
        self.assertRegex(bad.stdout, r'reported numbers \(\d+ entries\): \d+ failure')

    def test_extra_file_fails(self):
        with Copy() as sup:
            (sup / 'extensions/unlisted.json').write_text('{}')
            self.assertEqual(list(rv.integrity(sup, DATA)), ['extra file     manuscript/supplement/extensions/unlisted.json'])

    def test_changed_number_fails_by_digest_and_by_value(self):
        with Copy() as sup:
            edit_json(sup / 'extensions/paired_budget_summary.json', lambda d: d['primary'].update(point=0.99))
            self.assertIn('changed file   manuscript/supplement/extensions/paired_budget_summary.json', list(rv.integrity(sup, DATA)))
            rehash(sup, 'extensions/paired_budget_summary.json')
            self.assertEqual(list(rv.integrity(sup, DATA)), [])
            self.assertTrue(any(m.startswith('paired/difference_pp:') for m in rv.reported(sup)))

    def test_changed_empty_set_count_fails(self):
        with Copy() as sup:
            name = 'extensions/application_sensitivity_B50.json'
            edit_json(sup / name, lambda rows: rows[-1]['finite_sample']['N'].update(empty_sets=1)); rehash(sup, name)
            self.assertEqual([m.split(':')[0] for m in rv.reported(sup)], ['uci/sensitivity_B50_empty_sets'])

    def test_changed_table_cell_source_fails(self):
        with Copy() as sup:
            name = 'extensions/sphere_summary.json'
            edit_json(sup / name, lambda d: d['results'][0]['finite_sample']['200']['N'].update(mean_width=1.6)); rehash(sup, name)
            failed = {m.split(':')[0] for m in rv.reported(sup)}
            self.assertIn('S6/primary/B200/width_N', failed); self.assertIn('sphere/width_N_B200', failed)

    def test_corrupted_value_file_fails(self):
        with Copy() as sup:
            edit_json(sup / 'extension_numbers.json', lambda d: d['e7'].update(theta=d['e7']['theta'] + 1e-3))
            self.assertTrue(any(m.startswith('extension_numbers.json') for m in rv.value_files(sup, rv.recompute_all(sup))))

    def test_full_check_passes(self):
        ok = run(SUP)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn('All checks passed', ok.stdout)


if __name__ == '__main__':
    unittest.main()
