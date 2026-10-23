"""Deterministic unit fixtures for the overlapping-window experiment; no experiment is launched."""
import math
import unittest
from unittest.mock import patch
import numpy as np
from c11_upgrade import e8
from c11_upgrade.addendum_core import rng


class E8Fixtures(unittest.TestCase):
    def test_overlapping_windows_share_one_record(self):
        x=np.array([0,1,0,0,1,1]);y=1-x
        xw,yw=e8.consecutive_windows(x,y)
        self.assertEqual(xw.shape,(4,3))
        np.testing.assert_array_equal(xw[:-1,1:],xw[1:,:-1])
        np.testing.assert_array_equal(yw[:,1],y[1:-1])
        self.assertTrue(np.shares_memory(x,xw))

    def test_record_uses_actual_previous_state(self):
        # Deterministic alternation, detector recognizes only a transition 0 -> 1.
        x,y=e8.generate_record(np.random.default_rng(990001),20,stay=0,detect=(1,0),phi=0)
        self.assertTrue(np.all(x[1:]!=x[:-1]))
        np.testing.assert_array_equal(y,x)
        # Persistent positive process chooses detect[1], not detect[0].
        x,y=e8.generate_record(np.random.default_rng(990002),10,prevalence=1,rho=1,detect=(1,0))
        np.testing.assert_array_equal(x,np.ones(10))
        np.testing.assert_array_equal(y,np.zeros(10))

    def test_asymmetric_initial_stationary_probability_and_transitions(self):
        class RecordedRandom:
            def __init__(self): self.values=iter([.19,.75,.1,.79,.99,.05,.1])
            def random(self): return next(self.values)
        # m=.2 -> initial1; P11=.68 => first0; P01=.08 -> next0; final1.
        x,y=e8.generate_record(RecordedRandom(),3,prevalence=.2,rho=.6,detect=(.95,.2),phi=.02)
        np.testing.assert_array_equal(x,[0,0,1]);np.testing.assert_array_equal(y,[0,0,1])

    def test_priority_coupling_and_exact_thin_spacing(self):
        p=np.linspace(1,0,100)
        for k in (3,10):
            eligible=np.arange(0,100,k)
            selected=e8.priority_select(p,eligible,5)
            np.testing.assert_array_equal(np.diff(eligible),np.full(len(eligible)-1,k))
            np.testing.assert_array_equal(selected,eligible[-5:][::-1])
        np.testing.assert_array_equal(e8.priority_select(np.ones(10),np.array([8,2,5]),2),[2,5])
        with self.assertRaises(ValueError): e8.priority_select(p,np.arange(0,100,10),11)

    def test_block_empty_excluded_and_equal_weight_not_row_weight(self):
        obs=np.zeros((12,3),int);ids=np.repeat(np.arange(3),4)
        indices=np.array([0,1,2,8]);gold=np.array([0,0,0,1])
        limits,d=e8.block_moment_limits(obs,gold,obs[indices],indices,ids,.04,'N')
        self.assertEqual(d['audit_block_counts'],[3,0,1])
        self.assertEqual(d['nonempty_audit_blocks'],2)
        self.assertEqual(d['centers']['m'],.5)
        r=math.sqrt(math.log(2/(.04/11))/(2*2))
        self.assertAlmostEqual(d['radii']['m'],r)
        self.assertEqual(limits['m'],(max(0,.5-r),min(1,.5+r)))
        self.assertAlmostEqual(d['radii']['o0'],math.sqrt(math.log(2/(.04/11))/6))

    def test_information_error_allocation_and_hand_radius(self):
        obs=np.zeros((200,3),int);idx=np.arange(200);gold=idx%2
        for info,count in [('M',9),('N',11),('F',24)]:
            limits,d=e8.block_moment_limits(obs,gold,obs,idx,idx,.05,info)
            self.assertEqual(len(limits),count)
            self.assertAlmostEqual(d['delta'],.05/count)
            self.assertAlmostEqual(d['radii']['o0'],math.sqrt(math.log(2*count/.05)/400))
            lo,hi=limits['o0'];self.assertAlmostEqual(lo,1-d['radii']['o0']);self.assertEqual(hi,1)

    def test_block_calibration_zero_fp_is_not_zero_bound(self):
        indices=np.array([0,1,8]);ids=np.repeat(np.arange(3),4)
        upper,d=e8.block_upper(np.ones(3),np.ones(3),indices,ids)
        self.assertEqual(d['calibration_k'],0)
        self.assertEqual(d['calibration_block_counts'],[2,0,1])
        self.assertAlmostEqual(upper,min(1,math.sqrt(math.log(100)/4)))
        self.assertGreater(upper,0)
        upper,d=e8.block_upper([1,1,0],[1,1,1],indices,ids)
        self.assertEqual(d['center'],.5)

    def test_block_rejects_alignment_empty_duplicates_and_invalid(self):
        obs=np.zeros((4,3),int);ids=np.array([0,0,1,1])
        with self.assertRaises(ValueError): e8.block_moment_limits(obs,[],np.empty((0,3)),[],ids)
        with self.assertRaises(ValueError): e8.block_moment_limits(obs,[1,1],obs[[0,0]],[0,0],ids)
        with self.assertRaises(ValueError): e8.block_moment_limits(obs,[1],np.ones((1,3)),[0],ids)
        with self.assertRaises(ValueError): e8.block_upper([1],[0],[4],ids)
        with self.assertRaises(ValueError): e8.block_upper([1],[0],[0],ids,0)
        with self.assertRaises(ValueError): e8.block_upper([1],[0],[0],[float('nan')])

    def test_rng_is_design_and_cohort_separated(self):
        a=rng(3,0,990001,0).random(20)
        np.testing.assert_array_equal(a,rng(3,0,990001,0).random(20))
        self.assertFalse(np.array_equal(a,rng(3,0,990001,2).random(20)))
        self.assertFalse(np.array_equal(a,rng(4,0,990001,0).random(20)))

    def test_frozen_condition_order_and_seed_guard(self):
        g=e8.condition_grid()
        self.assertEqual([(c['stay'],c['phi']) for c in g[:6]],[(.2,0),(.2,.02),(.5,0),(.5,.02),(.8,0),(.8,.02)])
        self.assertEqual(g[6]['prevalence'],.2)
        for seed in (440000,441001,450001):
            with self.assertRaises(ValueError): e8.run_replicate(g[0],seed)

    def test_incomplete_grid_never_selects_method(self):
        rows=[dict(condition_id=0,method=m,seed=990001,width=1.,contains_population=True,
                   covers_truth=True,status='ok') for m in e8.METHODS]
        result=e8.summarize(rows,bootstrap_reps=3)
        self.assertFalse(result['complete_grid'])
        self.assertEqual(result['h8a'],'incomplete_protocol')
        self.assertEqual(result['selection']['method'],'application_only')
        self.assertAlmostEqual(result['groups'][0]['paired_width_difference'],0)

    def test_summary_multiplicity_undefined_and_failure_denominator(self):
        rows=[dict(condition_id=0,method=m,seed=990001,width=2.,contains_population=False,
                   covers_truth=False,status='solver_failure') for m in e8.METHODS]
        rows.append(dict(condition_id=0,method='block_200',seed=990002,width=2.,contains_population=None,
                         covers_truth=False,status='empty'))
        result=e8.summarize(rows,smoke=True,bootstrap_reps=2)
        groups={r['method']:r for r in result['groups']}
        self.assertAlmostEqual(groups['naive_overlap']['mc_error'],.05/6)
        self.assertAlmostEqual(groups['thin_10']['mc_error'],.05/12)
        self.assertEqual(groups['block_200']['valid_reference_n'],1)
        self.assertEqual(groups['block_200']['undefined_reference_n'],1)
        self.assertEqual(groups['block_200']['solver_failure_n'],1)
        self.assertEqual(groups['block_200']['mean_width'],2)
        self.assertEqual(result['h8a'],'NOT_EVALUATED_test_or_smoke')

    def test_runner_shares_audit_masks_and_independent_calibration_stream(self):
        # Fixed zero fixture and stub solver: exercises wiring without science.
        condition=e8.condition_grid()[1]
        record=np.zeros(10002,dtype=int)
        reference=dict(theta=1.,f=0.,intervals={'N':dict(status='ok',lower=1.,upper=1.)})
        def episodes(random,n,**kwargs):
            return np.zeros((n,3),int),np.zeros((n,3),int)
        with patch.object(e8,'generate_record',return_value=(record,record)) as gen, \
             patch.object(e8,'sample_episodes',side_effect=episodes), \
             patch.object(e8,'lp_interval',return_value=dict(status='ok',lower=-1.,upper=1.,residual=0.)):
            rows=e8.run_replicate(condition,990001,reference)['metrics']
        self.assertEqual(gen.call_count,2)
        rows={r['method']:r for r in rows}
        self.assertEqual(rows['thin_3']['observation_n'],3334)
        self.assertEqual(rows['thin_10']['observation_n'],1000)
        for method in ('naive_overlap','block_50','block_200'):
            self.assertEqual(rows[method]['diagnostics']['audit_indices'],rows['naive_overlap']['diagnostics']['audit_indices'])
            self.assertEqual(rows[method]['audit_n'],500)
            self.assertEqual(rows[method]['calibration']['calibration_n'],500)
        self.assertNotEqual(rows['naive_overlap']['diagnostics']['audit_indices'],rows['naive_overlap']['calibration']['calibration_indices'])
        self.assertTrue(all(v%10==0 for v in rows['thin_10']['diagnostics']['audit_indices']))

    def test_complete_synthetic_summary_tie_break_and_duplicate_block(self):
        # Hand-authored success flags, not generated experimental outcomes.
        rows=[dict(condition_id=cid,method=m,seed=seed,width=1.,contains_population=True,
                   covers_truth=True,status='ok')
              for cid in range(7) for m in e8.METHODS for seed in range(440001,441001)]
        with patch.object(e8,'cp_interval',return_value=(.94,.99)):
            out=e8.summarize(rows,bootstrap_reps=1)
            self.assertTrue(out['complete_grid'])
            self.assertEqual(out['selection']['method'],'thin_10')
            self.assertEqual(out['h8b']['block_200'],'large_loss_excluded')
            rows.append(rows[0].copy())
            out=e8.summarize(rows,bootstrap_reps=1)
            self.assertFalse(out['complete_grid'])
            self.assertEqual(out['selection']['method'],'application_only')

    def test_decision_refutation_precedes_loss_exclusion(self):
        self.assertEqual(e8._decision([(.935,.944)]*6),'refutation_signal')
        self.assertEqual(e8._decision([(.94,.99)]*6),'large_loss_excluded')
        self.assertEqual(e8._decision([(.94,.99)]*5),'inconclusive')


if __name__=='__main__': unittest.main()
