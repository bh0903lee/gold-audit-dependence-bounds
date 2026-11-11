import copy
import contextlib
import io
import sys
import os
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from c11_followup import p2, p2_cli, p2_safety
from c11_followup.p2_safety import check_output, REPOSITORY
from c11_upgrade.population import periodic_law


class P2Tests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='paired_budget_root_')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        patcher=patch.object(p2_safety,'OUTPUT_ROOT',self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_known_zero_and_shared_samples(self):
        for law_seed in p2.KNOWN_ZERO:
            spec = p2.law_specs()[law_seed-310041]
            rows = p2.run_replicate(spec, 200, 990001, law=periodic_law(law_seed, 3, 4))
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(r['sample_hashes'] == rows[0]['sample_hashes'] for r in rows))
            for info in ('M','N','F'):
                oracle = next(r for r in rows if r['arm']=='oracle_f_05' and r['method']==info)
                known = next(r for r in rows if r['arm']=='known_zero' and r['method']==info)
                self.assertEqual({k:v for k,v in oracle.items() if k!='arm'},
                                 {k:v for k,v in known.items() if k!='arm'})

    def test_seed_gate(self):
        spec = p2.law_specs()[0]
        for seed in (480000,480201,990001,True):
            with self.assertRaises(ValueError):
                p2.run_replicate(spec,200,seed,confirmatory=True)
        with self.assertRaises(ValueError):
            p2.run_replicate(spec,200,480001,law=periodic_law(990001,3,4))

    def test_fallback_width_and_reference_separation(self):
        spec=p2.law_specs()[0]
        fake={i:dict(status='solver_failure',lower=None,upper=None) for i in ('M','N','F')}
        with patch.object(p2,'solve_information',return_value=fake):
            rows=p2.run_replicate(spec,200,990001,law=periodic_law(990001,3,4))
        self.assertTrue(all(r['width']==2 and r['fallback_used'] and r['solver_failure'] for r in rows))
        for info in ('M','N','F'):
            values={(r['population_lower'],r['population_upper']) for r in rows if r['method']==info}
            self.assertEqual(len(values),1)
        self.assertTrue(all(r['fallback_contains_population'] is None for r in rows))

    def test_ratio_of_means_not_mean_ratios(self):
        self.assertAlmostEqual(p2.ratio_of_mean_widths([[2,1,0],[1,1,0]])['value'],1/3)
        self.assertEqual(p2.ratio_of_mean_widths([[0,0,0]])['undefined_reason'],'zero_denominator')
        self.assertEqual(p2.ratio_of_mean_widths([[np.nan,0,0]])['undefined_reason'],'invalid_widths')

    def test_paired_indices_all_arms_information(self):
        a=np.arange(2*3*4*3).reshape(2,3,4,3)
        idx=np.array([[2,0,2],[1,1,0]])
        b=p2.paired_resample(a,idx)
        for law in range(2):
            for rep in range(3):
                self.assertTrue(np.array_equal(b[law,rep],a[law,idx[law,rep]]))

    def test_fixed_panel_bootstrap_constant_ratio(self):
        a=np.ones((4,5,4,3)); a[:,:,0,1]=.5; a[:,:,2,1]=.75
        result=p2.paired_bootstrap(a,20,990001)
        self.assertEqual(result['point'],.25)
        self.assertEqual(result['ci_95'],[.25,.25])
        a[0,:,0,0]=0
        result=p2.paired_bootstrap(a,20,990001)
        self.assertEqual(result['defined_laws'],3)
        self.assertEqual(result['status'],'not_evaluable')
        self.assertEqual(result['point'],.25)

    def test_undefined_resample_not_silently_dropped(self):
        a=np.ones((1,2,4,3));a[0,0,:,0]=0
        result=p2.paired_bootstrap(a,200,990001)
        self.assertEqual(result['status'],'not_evaluable')
        self.assertEqual(result['reason'],'undefined_bootstrap_ratio')

    def test_undefined_diagnostic_arm_blocks_fixed_panel(self):
        a=np.ones((2,3,4,3)); a[0,:,3,0]=0
        self.assertEqual(p2.paired_bootstrap(a,20,990001)['status'],'not_evaluable')

    def test_unexpected_error_retains_failed_state(self):
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            out=Path(tmp)
            quota=type('Quota',(),{'check':lambda self:None})()
            with patch.object(p2_cli.p2,'CONFIRM_SEEDS',(990001,)), patch.object(p2_cli,'periodic_law',return_value=None), patch.object(p2_cli,'population_references',return_value={}), patch.object(p2_cli.p2,'run_replicate',side_effect=ValueError('injected fixture failure')):
                with self.assertRaises(ValueError):
                    p2_cli.full(out,quota,{'resources':{'memory_limit_kib':4194304}})
            state=json.loads((out/'run_state.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertEqual(state['h1'],'not_evaluable')

    def test_timing_hides_downstream_output_and_error_payload(self):
        sentinel='SYNTHETIC_SENSITIVE_MARKER_718'
        stdout,stderr=io.StringIO(),io.StringIO()
        def noisy(*args):
            print(sentinel)
            print(sentinel,file=sys.stderr)
            os.write(1,sentinel.encode())
            os.write(2,sentinel.encode())
            raise ValueError(sentinel)
        with patch.object(p2_cli,'_smoke',side_effect=noisy):
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                with self.assertRaises(RuntimeError) as caught:
                    p2_cli.smoke(None,None)
        self.assertEqual(stdout.getvalue(),'')
        self.assertEqual(stderr.getvalue(),'')
        self.assertNotIn(sentinel,str(caught.exception))
        self.assertIn('ValueError',str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_incomplete_grid_blocks_decision(self):
        result=p2.summarize([],confirmatory=True)
        self.assertEqual(result['h1'],'not_evaluable')
        self.assertFalse(result['full_grid_complete'])

    def test_complete_grid_metadata_only(self):
        # No law simulation or confirmatory-seed RNG: fabricated metadata validates gating only.
        rows=[]
        for s in p2.law_specs():
            for b in p2.BUDGETS:
                for seed in p2.CONFIRM_SEEDS:
                    for arm in (*p2.ARMS,*(('known_zero',) if s['law_seed'] in p2.KNOWN_ZERO else ())):
                        for info in ('M','N','F'):
                            rows.append({**s,'budget':b,'seed':seed,'arm':arm,'method':info,
                                'confirmatory':True,'n_obs':10000,'audit_n':b,'calibration_n':b,
                                'condition_id':3*s['law_index']+p2.BUDGETS.index(b),
                                'sample_hashes':dict(observation='a'*64,audit='b'*64,calibration='c'*64)})
        self.assertEqual(len(rows),291600)
        self.assertTrue(p2.complete_grid(rows))
        self.assertFalse(p2.complete_grid(rows[:-1]))
        rows[-1]=rows[0]
        self.assertFalse(p2.complete_grid(rows))
        rows[1]['sample_hashes']['audit']='d'*64
        self.assertFalse(p2.complete_grid(rows))

    def test_output_boundaries(self):
        with self.assertRaises(ValueError):check_output(REPOSITORY/'no')
        with self.assertRaises(ValueError):check_output('/tmp/no')
        with self.assertRaises(ValueError):check_output(self.root)
        relative=os.path.relpath(self.root/'relative_child')
        self.assertEqual(check_output(relative),self.root/'relative_child')
        p=self.root/'child'; p.mkdir()
        self.assertEqual(check_output(p),p)
        (p/'link').symlink_to(REPOSITORY,target_is_directory=True)
        with self.assertRaises(ValueError):check_output(p/'link'/'any')
        with self.assertRaises(ValueError):check_output(p)


if __name__=='__main__':unittest.main()
