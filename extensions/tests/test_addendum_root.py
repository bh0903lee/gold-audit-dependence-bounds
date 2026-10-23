"""CLI integration and negative-path tests; they run no experiments and read no saved results."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from c11_upgrade import e8, e7, addendum_cli as cli


class RootAddendumTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='root_fixture_')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve()
        patcher=patch.object(cli,'OUTPUT_ROOT',self.base)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_actual_plan_checks_frozen_bytes_and_executes_no_replicates(self):
        with patch.object(e8,'run_replicate',side_effect=AssertionError('No science in plan')):
            summary=cli.plan(self.base/'plan')
        self.assertEqual(summary['executed_replicates'],0)
        self.assertEqual(summary['e9']['endpoint_lps'],288000)
        manifest=json.loads((self.base/'plan/manifest.json').read_text())
        self.assertFalse(manifest['scientific_run'])
        self.assertFalse(manifest['run_complete'])
        self.assertIn('tests/test_addendum_root.py',manifest['hashes'])
        self.assertEqual(manifest['output_hashes']['summary.json'],cli.sha256(self.base/'plan/summary.json'))
        with self.assertRaises(ValueError): cli.plan(self.base/'plan')

    def test_paths_outside_and_symlink_are_rejected(self):
        for path in (cli.OUTPUT_ROOT,cli.ROOT/'outputs/new'):
            with self.assertRaises(ValueError): cli.output_path(path)
        (self.base/'escape').symlink_to(cli.ROOT,target_is_directory=True)
        with self.assertRaises(ValueError): cli.output_path(self.base/'escape/new')

    def test_baseline_hash_mismatch_is_rejected_before_output(self):
        real=cli.sha256
        def changed(path):
            return '0'*64 if Path(path)==cli.ROOT/'src/c11_upgrade/population.py' else real(path)
        with patch.object(cli,'sha256',changed):
            with self.assertRaisesRegex(ValueError,'Protected baseline'): cli.start_run(self.base/'blocked','plan_only')
        self.assertFalse((self.base/'blocked').exists())

    def test_e8_selection_rejects_plan_root_and_outside_paths(self):
        cli.plan(self.base/'plan')
        with self.assertRaises(ValueError): cli.e8_selection(self.base/'plan')
        with self.assertRaises(ValueError): cli.e8_selection(cli.ROOT/'outputs')
        with self.assertRaises(ValueError): cli.e8_selection(cli.OUTPUT_ROOT)

    def test_record_generator_matches_manual_state_history(self):
        class Uniforms:
            def __init__(self,values): self.values=iter(values)
            def random(self): return next(self.values)
        # Initial X[-1]=1; X=(0,1,1); Y uses previous X, not current X index.
        x,y=e8.generate_record(Uniforms([.1,.9,.01,.2,.7,.1,.3]),length=3,stay=.2,phi=.02)
        self.assertEqual(x.tolist(),[0,1,1]);self.assertEqual(y.tolist(),[1,1,0])
        # m=.2,rho=.6 gives p01=.08,p10=.32. Same history requires genuine asymmetric transition.
        x,y=e8.generate_record(Uniforms([.15,.7,.3,.07,.8,.67,.1]),length=3,prevalence=.2,rho=.6,phi=.02)
        self.assertEqual(x.tolist(),[0,1,1]);self.assertEqual(y.tolist(),[0,1,1])

    def test_rng_stream_matches_explicit_seedsequence(self):
        from c11_upgrade.addendum_core import rng
        expected=np.random.default_rng(np.random.SeedSequence([20261014,11,3,6,990002,2])).random(10)
        np.testing.assert_array_equal(rng(3,6,990002,2).random(10),expected)

    def test_e8_selection_recomputed_from_synthetic_complete_grid(self):
        # Artificial bookkeeping records, not simulated experimental outcomes.
        rows=[]
        for cid in range(7):
            for method in e8.METHODS:
                for seed in range(440001,441001):
                    rows.append(dict(condition_id=cid,method=method,seed=seed,contains_population=True,
                                     covers_truth=True,width=1.,status='ok',fallback_used=False,
                                     artificial_test_record=True))
        summary=e8.summarize(rows,bootstrap_reps=1)
        out=self.base/'synthetic_selection';out.mkdir()
        (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cli.write_json(out/'summary.json',summary)
        cli.write_json(out/'manifest.json',dict(kind='e8_full',run_complete=True,
            hashes={name:cli.sha256(cli.ROOT/name) for name in
                    (set(cli.validate_config()['baseline_hashes']) | {
                     'src/c11_upgrade/e8.py','src/c11_upgrade/addendum_core.py',
                     'src/c11_upgrade/addendum_cli.py','configs/addendum_e7_e9.json'})},
            output_hashes={name:cli.sha256(out/name) for name in ('rows.jsonl','summary.json')},
            synthetic_test_only=True))
        selected=cli.e8_selection(out)
        self.assertEqual(selected['method'],'thin_10')
        self.assertFalse(selected['coverage_certification'])
        (out/'rows.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'bytes differ'): cli.e8_selection(out)

    def test_e7_real_lp_fixture_keeps_reference_undefined_but_point_separate(self):
        # Repeated nonstationary empirical pattern, no real data or science seed.
        frames=[]
        for i in range(120):
            v=[0,0,0] if i%2==0 else [0,0,1]
            frames.append(dict(x=v,y=v,unit='room',day=str(i//60),times=[60*i,60*i+60,60*i+120],block_id=i//20))
        ref=e7.empirical_reference(frames)
        self.assertTrue(all(r['status']=='empty' for r in ref['intervals'].values()))
        frame=dict(frames=frames,method='application_only',lag_multiplier=1,status='ready',unit_days=2)
        rep=e7.run_replicate(frame,50,990001,ref)
        self.assertEqual(len(rep['metrics']),3)
        self.assertTrue(all(r['contains_population'] is None for r in rep['metrics']))
        self.assertTrue(all(isinstance(r['covers_truth'],bool) for r in rep['metrics']))
        summary=e7.summarize(rep['metrics'],frame,ref,dict(status='dependent_inference_required'),False)
        self.assertEqual(summary['application_criterion'],'NOT_EVALUATED_test_fixture')


if __name__=='__main__': unittest.main()
