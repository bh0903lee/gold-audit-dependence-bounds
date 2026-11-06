"""Independent oracles; only hand-built rational laws are evaluated."""
from fractions import Fraction as Q
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog, brentq
from scipy.stats import binom
from c11_upgrade import addendum_core as core, e9
from c11_upgrade.population import lp_interval, INFO


def rational_fixture():
    # Build stationarity by uniformly choosing the origin of each finite cycle.
    canonical={(0,0):0,(1,0):1,(1,1):2,(0,1):3}
    cycles=[([(0,0),(1,0),(1,1),(0,1)],Q(1,2)),
            ([(1,1),(1,1),(0,0)],Q(1,3)),
            ([(1,0),(0,0),(1,0),(1,1),(0,1)],Q(1,6))]
    law=np.zeros((4,4,4),dtype=object);mom={f'o{k}':Q(0) for k in range(8)}
    mom.update({f'u{k}':Q(0) for k in range(16)});mom.update(m=Q(0),a=Q(0),b=Q(0),fp=Q(0))
    for cycle,weight in cycles:
        for start in range(len(cycle)):
            triple=[cycle[(start+j)%len(cycle)] for j in range(3)]
            x=[v[0] for v in triple];y=[v[1] for v in triple];w=weight/len(cycle)
            law[tuple(canonical[v] for v in triple)]+=w
            o=4*y[0]+2*y[1]+y[2]
            mom[f'o{o}']+=w;mom[f'u{8*x[1]+o}']+=w
            mom['m']+=w*x[1];mom['a']+=w*y[0]*x[1];mom['b']+=w*x[1]*y[2]
            mom['fp']+=w*(1-x[1])*y[1]
    return law,mom


def independent_lp(mom,eps,information):
    # Variable order is six binary coordinates, unlike the production Z order.
    words=np.array(list(product((0,1),repeat=6)))
    x=words[:,[0,2,4]];y=words[:,[1,3,5]];z=2*x+y
    obj=(2*x[:,0]-1)*(2*x[:,1]-1)
    ae=[np.ones(64)];be=[1.]
    for a,b in product(range(4),repeat=2):
        ae.append(((z[:,0]==a)&(z[:,1]==b)).astype(int)-((z[:,1]==a)&(z[:,2]==b)).astype(int));be.append(0.)
    oid=y@np.array([4,2,1]);audit=8*x[:,1]+oid
    fs={f'o{k}':(oid==k).astype(int) for k in range(8)}
    fs.update({f'u{k}':(audit==k).astype(int) for k in range(16)})
    fs.update(m=x[:,1],a=y[:,0]*x[:,1],b=x[:,1]*y[:,2])
    keys=[f'o{k}' for k in range(8)]+({'M':['m'],'N':['m','a','b'],'F':[f'u{k}' for k in range(16)]}[information])
    for key in keys: ae.append(fs[key]);be.append(float(mom[key]))
    au=np.array([(1-x[:,1])*y[:,1]]);bu=np.array([float(eps)])
    answers=[]
    for sign in (1,-1):
        solve=linprog(sign*obj,A_eq=np.array(ae),b_eq=be,A_ub=au,b_ub=bu,bounds=(0,None),method='highs',options={'presolve':False})
        if not solve.success: raise AssertionError(solve.message)
        answers.append(float(obj@solve.x))
    return answers


class IndependentE9(unittest.TestCase):
    def test_distinct_binary_encoding_lp_oracle(self):
        law,mom=rational_fixture();limits={k:(v,v) for k,v in mom.items() if k!='fp'}
        before=dict(INFO)
        for eps in (mom['fp'],mom['fp']+Q(1,10),Q(1)):
            for info in ('M','N','F'):
                want=independent_lp(mom,eps,info)
                got=lp_interval(core.make_problem(limits,eps,info))
                self.assertEqual(got['status'],'ok')
                np.testing.assert_allclose([got['lower'],got['upper']],want,atol=2e-8,rtol=0)
        self.assertEqual(before,INFO)
        ref=core.population_references(law)
        self.assertAlmostEqual(ref['f'],float(mom['fp']))
        self.assertAlmostEqual(ref['stationarity_residual'],0)

    def test_cp_limits_from_binomial_tail_inversion(self):
        obs=np.array([[0,0,0],[1,0,1],[0,1,0],[0,0,0],[1,1,1]])
        gold=np.array([0,1,1,0,1]);audit=obs.copy()
        for info,count in [('M',9),('N',11),('F',24)]:
            got=core.information_limits(obs,gold,audit,info,.04)
            self.assertEqual(len(got),count)
            # Exact equal-tail inversion with k=2,n=5, independent of beta code.
            tail=.04/count/2
            lo=brentq(lambda p:binom.sf(1,5,p)-tail,0,1)
            hi=brentq(lambda p:binom.cdf(2,5,p)-tail,0,1)
            np.testing.assert_allclose(got['o0'],[lo,hi],atol=1e-11,rtol=0)

    def test_sampler_passes_all64_probabilities_and_bit_mapping(self):
        raw=np.arange(1,65,dtype=float);raw/=raw.sum();law=raw.reshape(4,4,4)
        class ChoiceRecorder:
            def choice(self,n,size,p):
                if n!=64 or size!=64: raise AssertionError('Unexpected categorical frame')
                np.testing.assert_allclose(p,raw,atol=1e-15)
                return np.arange(64)
        x,y=e9.sample_words(ChoiceRecorder(),law,64)
        expected=[(0,0),(1,0),(1,1),(0,1)]
        for row,triple in enumerate(product(range(4),repeat=3)):
            self.assertEqual(x[row].tolist(),[expected[j][0] for j in triple])
            self.assertEqual(y[row].tolist(),[expected[j][1] for j in triple])

    def test_metadata_excludes_disclosed_bank_without_generation(self):
        with patch.object(e9,'periodic_law',side_effect=AssertionError('Forbidden bank generation')):
            specs=list(e9.law_specs())
            self.assertEqual({s['law_seed'] for s in specs if s['bank']=='C'},set(range(310041,310081)))
            self.assertEqual({s['law_seed'] for s in specs if s['bank']=='A'},set(range(300001,300041)))
            for bad in (310001,310020,310040):
                with self.assertRaises(ValueError):
                    e9.run_replicate({**specs[40],'law_seed':bad},200,990001,law=np.ones((4,4,4))/64)

    def test_bank_c_zero_mass_uses_calibration_and_shared_inputs(self):
        spec=list(e9.law_specs())[40];p=np.zeros((4,4,4));p[0,0,0]=1
        observed=[]
        def solve(oy,x1,ay,epsilon,alpha):
            observed.append((oy.copy(),x1.copy(),ay.copy(),epsilon,alpha))
            return {info:dict(status='ok',lower=-1.,upper=1.) for info in ('M','N','F')}
        reference=dict(theta=1.,f=0.,intervals={info:dict(status='ok',lower=1.,upper=1.) for info in ('M','N','F')})
        with patch.object(e9,'solve_information',side_effect=solve),patch.object(e9,'periodic_law',side_effect=AssertionError('No banks')):
            result=e9.run_replicate(spec,200,990001,law=p,reference=reference,n_obs=11)
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0][4],.04)
        self.assertAlmostEqual(observed[0][3],1-.01**(1/200))
        for row in result['metrics']:
            self.assertEqual(row['calibration_policy'],'learned_upper');self.assertEqual(row['calibration_n'],200)
            self.assertEqual(row['sample_hashes'],result['sample_hashes'])
            self.assertTrue(row['population_zero_width'])

    def test_empty_failure_reference_and_theta_are_distinct(self):
        ok=dict(status='ok',lower=-.2,upper=.3);empty=dict(status='empty',lower=None,upper=None)
        fail=dict(status='solver_failure',lower=None,upper=None)
        for raw in (empty,fail):
            got=core.interval_metrics(raw,ok,0)
            self.assertEqual(got['width'],2);self.assertFalse(got['contains_population'])
            self.assertFalse(got['covers_truth']);self.assertTrue(got['fallback_used'])
        self.assertIsNone(core.interval_metrics(fail,ok,0)['fallback_covers_truth'])
        self.assertTrue(core.interval_metrics(empty,ok,0)['fallback_covers_truth'])
        got=core.interval_metrics(ok,empty,0)
        self.assertIsNone(got['contains_population']);self.assertTrue(got['covers_truth'])

    def _grid(self):
        # Metadata/outcome fixture only. No random seed is consumed or law generated.
        rows=[]
        for spec in e9.law_specs():
            for bi,budget in enumerate((200,500,2000)):
                for method,width in [('M',1.),('N',.8),('F',.7)]:
                    for seed in range(450001,450201):
                        rows.append({**spec,'budget':budget,'method':method,'width':width,'seed':seed,
                            'condition_id':3*spec['law_index']+bi,'confirmatory':True,'n_obs':10000,'audit_n':budget,
                            'calibration_n':budget if spec['bank']=='C' else 0,'population_reference_defined':True,
                            'contains_population':True,'covers_truth':True})
        return rows

    def test_complete_missing_duplicate_and_reference_denominators(self):
        rows=self._grid()
        with patch.object(e9,'cp_interval',return_value=(.96,1.)):
            got=e9.summarize(rows,confirmatory=True)
            self.assertTrue(got['full_grid_complete']);self.assertEqual(got['h9b_success_laws'],80)
            self.assertEqual(got['h9b'],'prediction_consistent')
            self.assertEqual(e9.summarize(rows[:-1],True)['h9b'],'BLOCKED_incomplete_grid')
            self.assertEqual(e9.summarize(rows+[rows[0]],True)['h9b'],'BLOCKED_incomplete_grid')
            for row in rows:
                if row['law_index']==0 and row['budget']==500 and row['method']=='N':
                    row['population_reference_defined']=False;row['contains_population']=None
            got=e9.summarize(rows,confirmatory=True)
            self.assertEqual(got['h9b_total_laws'],80);self.assertEqual(got['h9b_success_laws'],79)
            self.assertEqual(got['h9b_comparable_laws'],79)
            self.assertFalse(got['all_population_references_defined'])
            self.assertEqual(got['h9b'],'BLOCKED_undefined_population_reference')
            self.assertEqual(got['h9a'],'NOT_EVALUATED')
            bad=[g for g in got['groups'] if g['law_index']==0 and g['budget']==500 and g['method']=='N'][0]
            self.assertEqual(bad['reference_undefined_n'],200);self.assertIsNone(bad['contains_population'])
            self.assertEqual(bad['contains_population_total_denominator'],0)


if __name__=='__main__': unittest.main()
