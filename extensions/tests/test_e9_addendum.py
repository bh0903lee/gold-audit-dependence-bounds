"""Implementation tests use synthetic fixtures only, never confirmation bank laws."""
import unittest
from unittest.mock import patch
from fractions import Fraction as Q
import numpy as np
from scipy.stats import beta
from c11_upgrade import addendum_core as core
from c11_upgrade import e9
from c11_upgrade.population import INFO, lp_interval


def fixture():
    p = np.zeros((4, 4, 4), dtype=object)
    p[0, 0, 0] = Q(1, 2); p[2, 2, 2] = Q(1, 2)
    return p


class CoreTests(unittest.TestCase):
    def test_stream_order(self):
        want = np.random.default_rng(np.random.SeedSequence([20261014,11,4,7,990001,2])).random(9)
        np.testing.assert_array_equal(want, core.rng(4,7,990001,2).random(9))
        with self.assertRaises(ValueError): core.rng(4,7,-1,2)

    def test_direct_budgets_against_beta(self):
        oy = np.array([[0,0,0],[1,1,1],[0,0,0],[1,0,1]])
        ay = oy[:3]; ax = np.array([0,1,1])
        for info, k in (("M",9),("N",11),("F",24)):
            limits = core.information_limits(oy,ax,ay,info,.04)
            self.assertEqual(len(limits),k)
            self.assertAlmostEqual(limits['o0'][0], beta.ppf(.04/k/2,2,3))
            self.assertAlmostEqual(limits['o0'][1], beta.ppf(1-.04/k/2,3,2))
        self.assertNotEqual(core.information_limits(oy,ax,ay,"M")["o0"],
                            core.information_limits(oy,ax,ay,"N")["o0"])

    def test_m_problem_does_not_mutate_info(self):
        before = dict(INFO)
        limits = {k:(0,1) for k in core.information_names("M")}; limits["m"]=(0,0)
        r = lp_interval(core.make_problem(limits,0,"M"))
        self.assertEqual(r['status'], 'ok'); self.assertAlmostEqual(r['lower'],1)
        self.assertAlmostEqual(r['upper'],1); self.assertEqual(INFO,before)

    def test_reference_nonstationary_not_projected(self):
        p = np.zeros((4,4,4)); p[0,0,2] = 1
        refs = core.population_references(p)
        self.assertEqual(refs['theta'],1); self.assertGreater(refs['stationarity_residual'],0)
        self.assertEqual(refs['intervals']['F']['status'],'empty')

    def test_reference_undefined_failure_and_empty(self):
        ok={'status':'ok','lower':-.5,'upper':.5}
        empty={'status':'empty','lower':None,'upper':None}
        failed={'status':'solver_failure','lower':None,'upper':None}
        m = core.interval_metrics(failed,ok,0)
        self.assertIs(m['contains_population'],False); self.assertIsNone(m['fallback_contains_population'])
        self.assertEqual(m['width'],2)
        m=core.interval_metrics(empty,ok,0)
        self.assertIs(m['contains_population'],False); self.assertIs(m['fallback_contains_population'],True)
        m=core.interval_metrics(ok,empty,0)
        self.assertIsNone(m['contains_population']); self.assertIs(m['covers_truth'],True)

    def test_validation(self):
        for p in (np.ones(64),np.zeros((3,3,3)),np.full(64,np.nan)):
            with self.assertRaises(ValueError): core.validate_law(p)
        with self.assertRaises(ValueError): core.information_names('N8')
        with self.assertRaises(ValueError): core.make_problem({k:(0,1) for k in core.information_names('M')},np.nan,'M')


class E9Tests(unittest.TestCase):
    def test_specs_metadata_and_ranges(self):
        with patch.object(e9,'periodic_law',side_effect=AssertionError('Never generate banks during tests')):
            specs=list(e9.law_specs())
        self.assertEqual(len(specs),80)
        self.assertEqual([s['law_seed'] for s in specs[:40]],list(range(300001,300041)))
        self.assertEqual([s['law_seed'] for s in specs[40:]],list(range(310041,310081)))
        self.assertEqual([s['law_index'] for s in specs],list(range(80)))

    def test_sampling_deterministic_law(self):
        p=np.zeros((4,4,4));p[3,1,2]=1
        x,y=e9.sample_words(core.rng(4,0,990001,0),p,4)
        np.testing.assert_array_equal(x,np.tile([0,1,1],(4,1)))
        np.testing.assert_array_equal(y,np.tile([1,0,1],(4,1)))

    def test_seed_and_bank_law_protection(self):
        spec=list(e9.law_specs())[0]
        with patch.object(e9,'periodic_law',side_effect=AssertionError('Never generate banks during tests')):
            for kwargs in ({'seed':450001,'law':fixture()}, {'seed':990001},
                           {'seed':990001,'confirmatory':True},
                           {'seed':450001,'confirmatory':True,'law':fixture()}):
                with self.assertRaises(ValueError): e9.run_replicate(spec,200,**kwargs)
            bad={**spec,'bank':'C','law_seed':310001,'alphabet':4}
            with self.assertRaises(ValueError): e9.run_replicate(bad,200,990001,law=fixture())

    def test_c_zero_fp_still_calibrates_same_samples(self):
        spec=list(e9.law_specs())[40]
        out=e9.run_replicate(spec,200,990001,law=fixture(),n_obs=40)
        rows=out['metrics']
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertTrue(row['f_zero']);self.assertEqual(row['calibration_n'],200)
            self.assertEqual(row['calibration_k'],0);self.assertGreater(row['epsilon'],0)
            self.assertEqual(row['sample_hashes'],rows[0]['sample_hashes'])
        self.assertAlmostEqual(rows[0]['epsilon'],1-.01**(1/200))
        s=e9.summarize(rows,confirmatory=True)
        self.assertFalse(s['full_grid_complete']);self.assertEqual(s['h9b'],'BLOCKED_incomplete_grid')
        self.assertEqual(len(s['comparisons']),240)
        self.assertEqual(e9.summarize(rows)['h9b'],'NOT_EVALUATED_fixture_only')

    def test_a_known_zero_and_degenerate_tags(self):
        p=np.zeros((4,4,4));p[0,0,0]=1
        out=e9.run_replicate(next(e9.law_specs()),200,990001,law=p,n_obs=30)
        for row in out['metrics']:
            self.assertEqual(row['calibration_n'],0);self.assertEqual(row['epsilon'],0)
            self.assertTrue(row['population_zero_width']);self.assertTrue(row['f_zero'])

    def test_full_grid_undefined_reference_blocks_verdict(self):
        # Synthetic row fixture: no generators, law endpoints, or confirmation
        # sampling are evaluated. This tests only the summary acceptance gate.
        rows = []
        for spec in e9.law_specs():
            for bi, budget in enumerate(e9.BUDGETS):
                for method in core.INFORMATION:
                    base = {**spec, 'budget': budget, 'condition_id': 3*spec['law_index']+bi,
                            'method': method, 'confirmatory': True, 'n_obs': 10000,
                            'audit_n': budget, 'calibration_n': budget if spec['bank']=='C' else 0,
                            'width': 1.5 if method=='M' else 1., 'covers_truth': True,
                            'population_reference_defined': spec['law_index'] < 60,
                            'contains_population': True if spec['law_index'] < 60 else None}
                    rows.extend({**base, 'seed': seed} for seed in e9.CONFIRM_SEEDS)
        out = e9.summarize(rows, confirmatory=True)
        self.assertTrue(out['full_grid_complete'])
        self.assertEqual(out['h9b_success_laws'], 60)
        self.assertFalse(out['all_population_references_defined'])
        self.assertEqual(out['h9b'], 'BLOCKED_undefined_population_reference')

    def test_zero_width_ratio_not_deleted(self):
        rows=[]
        spec=next(e9.law_specs())
        for method in core.INFORMATION:
            rows.append({**spec,'budget':500,'condition_id':1,'method':method,'seed':990001,
                         'width':0.,'population_reference_defined':True,'contains_population':True,
                         'covers_truth':True})
        s=e9.summarize(rows)
        r=next(r for r in s['comparisons'] if r['law_index']==0 and r['budget']==500)
        self.assertEqual(r['N_vs_M']['undefined_reason'],'zero_denominator')
        self.assertIsNone(r['N_vs_M']['relative_reduction'])
        self.assertEqual(s['h9b_success_laws'],0)


if __name__=='__main__':
    unittest.main()
