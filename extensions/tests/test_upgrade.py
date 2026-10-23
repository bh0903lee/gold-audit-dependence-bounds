"""Implementation checks; no confirmatory seed bank is consumed."""
import copy
from fractions import Fraction as Q
from itertools import product
import unittest
import numpy as np
from c11_upgrade import population as p
from c11_upgrade import inference as inf
from c11_upgrade.adapter import fixture,validate_dataset
from c11_upgrade.cli import d1_bank,validate_config


class PopulationTests(unittest.TestCase):
    def test_no_fp_recovery_and_relaxation(self):
        law=p.markov_law(stay=.8,phi=0); limits=p.law_limits(law)
        closed=p.closed_form(limits)
        for info in ("N","N6","N8","F"):
            zero=p.lp_interval(p.word_problem(limits,0,info))
            relaxed=p.lp_interval(p.word_problem(limits,Q(1,10),info))
            self.assertEqual(zero["status"],"ok")
            np.testing.assert_allclose([zero["lower"],zero["upper"]],closed,atol=1e-8)
            self.assertLessEqual(relaxed["lower"],zero["lower"]+1e-8)
            self.assertGreaterEqual(relaxed["upper"],zero["upper"]-1e-8)

    def test_four_state_target_and_stationarity(self):
        self.assertEqual(p.X[3],0)
        law=p.periodic_law(990001)
        words,rows,flow,c=p.features()
        self.assertTrue(np.all(flow@law.ravel()==0))
        x=p.X[words]
        average=(((2*x[:,:-1]-1)*(2*x[:,1:]-1)).sum(axis=1)/2)
        self.assertAlmostEqual(float(c@law.ravel()),float(average@law.ravel()),12)
        for m in (.2,.8):
            ref=inf.population_reference(dict(prevalence=m,rho=.6,phi=.1,stay=.5,detect=[.95,.2]))
            self.assertAlmostEqual(ref["f"],(1-m)*.1)
            self.assertAlmostEqual(ref["theta"],1-8*m*(1-m)*.4/2)

    def test_m8_full_and_exact_gluing(self):
        law=p.periodic_law(990002); limits=p.law_limits(law); h=p.full_h(law)
        pair=p.pair_from_moments(limits); result=p.lp_interval(pair,True)
        full=p.lp_interval(p.word_problem(limits,p.fp_mass(law),"F"))
        for side in ("lower","upper"): self.assertAlmostEqual(result[side],full[side],8)
        for sol,sign in zip(result["solutions"],(1,-1)):
            cert=p.exact_certificate(pair,sol,sign)
            self.assertTrue(cert["verified"],cert)
            k=np.array([Q(v) for v in cert["primal"]],dtype=object).reshape(4,4)
            checks=p.check_gluing(k,h,p.glue_triple(k,h),exact=True)
            self.assertTrue(all(checks.values()),checks)

    def test_l4_audit_and_invalid_inputs(self):
        law=p.periodic_law(990003,length=4); limits=p.law_limits(law,4)
        problem=p.word_problem(limits,p.fp_mass(law,4),"F",4)
        self.assertEqual(sum(k.startswith("u") for k in problem.labels),32)
        self.assertEqual(len(problem.c),256)
        for bad in (-.1,float("nan"),1.1):
            with self.assertRaises(ValueError): p.markov_law(phi=bad)
            with self.assertRaises(ValueError): p.word_problem(limits,bad,length=4)


class InferenceTests(unittest.TestCase):
    def test_cp_boundaries(self):
        self.assertAlmostEqual(inf.cp_upper(0,500),1-.01**(1/500),12)
        self.assertEqual(inf.cp_upper(500,500),1)
        self.assertEqual(inf.cp_interval(0,5)[0],0)
        for args in ((-1,5),(6,5),(0,0),(1.5,5),(1,5,1)):
            with self.assertRaises(ValueError): inf.cp_upper(*args)

    def test_sampled_moments_match_exact_generator(self):
        args=dict(stay=.8,phi=.1,detect=(.95,.2),prevalence=.2,rho=.6)
        x,y=inf.sample_episodes(np.random.default_rng(990001),100000,**args)
        counts=inf.sample_counts(y,x[:,1]); exact=p.law_limits(p.markov_law(**args))
        for k in ("m","a","b","j","cL","cR"):
            self.assertLess(abs(counts[k]/len(y)-float(exact[k][0])),.007)

    def test_shared_nesting_and_isolation(self):
        _,oy=inf.sample_episodes(np.random.default_rng(990001),1000,phi=.1)
        ax,ay=inf.sample_episodes(np.random.default_rng(990002),200,phi=.1)
        result=inf.shared_box_comparison(oy,ax[:,1],ay,.1)
        self.assertEqual(result["nesting_violations"],[])
        # The public inference API accepts only X1, never adjacent latent truth.
        with self.assertRaises(ValueError): inf.infer_intervals(oy,ax,ay,ax[:,1],ay[:,1],.05)

    def test_decision_priority_and_partial_grid(self):
        self.assertEqual(inf.coverage_decision([(.935,.944)]*9),"refutation_signal")
        self.assertEqual(inf.coverage_decision([(.94,.99)]*9),"large_loss_excluded")
        self.assertEqual(inf.coverage_decision([(.94,.99)]*8),"inconclusive")
        self.assertEqual(inf.summarize([])["coverage_decision"],"incomplete_protocol")


class GateTests(unittest.TestCase):
    def test_eight_contract_fixtures(self):
        rows,manifest=fixture()
        base=validate_dataset(rows,manifest)
        self.assertEqual(base["status"],"eligible_iid_validation")
        self.assertGreater(base["epsilon_upper"],0)
        self.assertFalse(base["source_hash_verified"])
        only=copy.deepcopy(rows); only[-1].pop("x0_gold"); only[-1].pop("x2_gold")
        self.assertEqual(validate_dataset(only,manifest)["status"],"application_only")
        selected=copy.deepcopy(rows); selected[1]["selection_probability"]=0
        self.assertEqual(validate_dataset(selected,manifest)["status"],"blocked_selection")
        time=copy.deepcopy(rows); time[0]["t2"]=3
        self.assertEqual(validate_dataset(time,manifest)["status"],"blocked_schema")
        overlap=copy.deepcopy(rows); overlap[1]["unit_id"]=overlap[0]["unit_id"]
        self.assertEqual(validate_dataset(overlap,manifest)["status"],"dependent_inference_required")
        gold=copy.deepcopy(rows); gold[1]["gold_source"]=""
        self.assertEqual(validate_dataset(gold,manifest)["status"],"blocked_gold")
        transport=copy.deepcopy(rows); transport[1]["population_id"]="different"
        self.assertEqual(validate_dataset(transport,manifest)["status"],"stationarity_unsupported")
        one=copy.deepcopy(rows); one[-1].pop("x2_gold")
        self.assertEqual(validate_dataset(one,manifest)["status"],"eligible_iid_validation")

    def test_unequal_selection_and_nonselfweighting(self):
        rows,manifest=fixture(); extra=copy.deepcopy(rows[1]); extra.update(episode_id="extra",unit_id="extra",selection_probability=.1)
        self.assertEqual(validate_dataset(rows+[extra],manifest)["status"],"blocked_selection")
        manifest.pop("self_weighting_sample")
        self.assertEqual(validate_dataset(rows,manifest)["status"],"blocked_selection")


class DesignTests(unittest.TestCase):
    def test_frozen_config_and_grid_counts(self):
        c=validate_config(); grid=inf.condition_grid()
        self.assertEqual(len(grid),29)
        total=sum(1000 if g["family"]=="primary" else 200 for g in grid)
        self.assertEqual(total,c["d2_full_replicates"])
        smoke=list(d1_bank(False))
        self.assertTrue(all(seed in (None,990001,990002,990003) for _,seed,_,_ in smoke))


if __name__=="__main__": unittest.main()
