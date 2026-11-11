import unittest,json,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import engine as e,analysis as a
from unittest.mock import patch
class AuditRegression(unittest.TestCase):
    def test_lp_failure_is_retained_without_losing_other_conditions(self):
        config=dict(seeds=[60704],stay=[.2],episodes=30,audit_budgets=[10,20])
        with tempfile.TemporaryDirectory() as td,patch.object(e,'bounds',side_effect=[e.InfeasibleConfidenceSet('infeasible'),[-1,1],[-1,1],[-.8,.8]]):
            rows=e.run(config,Path(td));self.assertEqual(len(rows),4);self.assertTrue(rows[0]['empty_confidence_set']);self.assertEqual((rows[0]['low'],rows[0]['high']),(-1.,1.));self.assertEqual(sum(r['empty_confidence_set'] for r in rows),1)
    def test_solver_status_is_classified(self):
        # Fault injection: only status 2 is an empty confidence set; limits and numerical failures stay failures.
        from types import SimpleNamespace as NS
        from topic_experiment import sixvar as s,w3w2
        y=np.random.default_rng(1).integers(2,size=(200,3));ax=np.ones(50,int);ay=np.ones((50,3),int)
        for module in (e,s):
            for status,cls in ((2,e.InfeasibleConfidenceSet),(1,e.SolverFailure),(3,e.SolverFailure),(4,e.SolverFailure)):
                fake=NS(success=False,status=status,message=f'injected {status}',x=None)
                with patch.object(module,'linprog',return_value=fake):
                    for method in ('word/exact_binomial','sixvar/exact_binomial','l1ball'):
                        if (module is e)!=(method=='word/exact_binomial'):continue
                        if cls is e.InfeasibleConfidenceSet:
                            row=w3w2.evaluate(method,y,ax,ay,.05,[0,0,0],0.,pop=[-.1,.1])
                            self.assertTrue(row['empty_confidence_set']);self.assertIn('injected 2',row['error'])
                        else:
                            with self.assertRaises(e.SolverFailure) as caught:w3w2.evaluate(method,y,ax,ay,.05,[0,0,0],0.,pop=[-.1,.1])
                            self.assertIn(f'status {status}',str(caught.exception))
        config=dict(seeds=[60704],stay=[.2],episodes=30,audit_budgets=[10])
        with tempfile.TemporaryDirectory() as td,patch.object(e,'bounds',side_effect=e.SolverFailure('LP solver failure (status 4): injected')):
            with self.assertRaises(e.SolverFailure):e.run(config,Path(td))
    def test_primary_budget_is_500(self):
        rows=[dict(seed=1,stay=.2,audits=b,method=m,width=w,covered=True) for b in [100,500] for m,w in [('marginal/hoeffding',1.),('neighbor/hoeffding',.5)]]
        p=a.analyze('C11',rows)['primary_analysis'];self.assertEqual(len(p),1);self.assertEqual(p[0]['audits'],500);self.assertEqual(p[0]['contrast'],'neighbor_vs_marginal_hoeffding')

if __name__=='__main__':unittest.main(verbosity=2)
