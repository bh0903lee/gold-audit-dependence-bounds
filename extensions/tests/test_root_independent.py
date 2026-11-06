"""Independent oracle checks for the population LPs, inference and data-contract checks.

The LP oracle uses binary integer encoding 2*X+Y rather than the production
alphabet, constructs its own flow/observation constraints, and uses only the
first adjacent spin product as its objective. Fixtures are software checks,
not a search over the experiment's law bank.
"""
import itertools
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from c11_upgrade import population as impl
from c11_upgrade import inference, adapter


def cyclic_fixture(cycles):
    p = np.zeros((4, 4, 4), dtype=object)
    for seq in cycles:
        for start in range(len(seq)):
            w = tuple(seq[(start + j) % len(seq)] for j in range(3))
            p[w] += F(1, len(cycles) * len(seq))
    return p


def oracle(law, epsilon, mode):
    w = np.asarray(list(itertools.product(range(4), repeat=3)))
    x, y = w // 2, w % 2
    to_production = np.array([0, 3, 1, 2])
    p = np.asarray([float(law[tuple(to_production[row])]) for row in w])
    ae, be = [np.ones(64)], [1.0]
    for a, b in itertools.product(range(4), repeat=2):
        ae.append(((w[:, 0] == a) & (w[:, 1] == b)).astype(float)
                  - ((w[:, 1] == a) & (w[:, 2] == b)).astype(float))
        be.append(0.0)
    obs = [(y == q).all(1).astype(float) for q in itertools.product(range(2), repeat=3)]
    features = [x[:, 1], y[:, 0] * x[:, 1], x[:, 1] * y[:, 2]]
    if mode == 'F':
        features = [q * x[:, 1] for q in obs]
    elif mode == 'N6':
        features += [x[:, 1] * y[:, 1]]
    for f in obs + features:
        ae.append(f)
        be.append(f @ p)
    fp = ((x[:, 1] == 0) & (y[:, 1] == 1)).astype(float)
    c = (2*x[:, 0]-1)*(2*x[:, 1]-1)
    answers = []
    for sign in (1, -1):
        sol = linprog(sign*c, A_eq=ae, b_eq=be, A_ub=[fp],
                      b_ub=[float(epsilon)], bounds=(0, None), method='highs')
        if not sol.success:
            raise AssertionError(sol.message)
        answers.append(float(c @ sol.x))
    return answers


class IndependentPopulationTests(unittest.TestCase):
    def test_m8_interface_does_not_require_full_audit_law(self):
        law=cyclic_fixture([[0,1,3,2,2],[3,0]])
        values=impl.law_limits(law)
        m8={k:values[k][0] for k in ['m','y','a','b','h','j','cL','cR']}
        got=impl.lp_interval(impl.pair_from_moments(m8))
        np.testing.assert_allclose([got['lower'],got['upper']],oracle(law,1,'F'),atol=1e-8,rtol=0)

    def test_binary_encoded_oracle_and_information_controls(self):
        cases = [cyclic_fixture([[0, 1, 2, 3]]),
                 cyclic_fixture([[0, 0, 1, 2, 2, 3], [3, 1, 0]]),
                 cyclic_fixture([[0], [1], [2], [3]])]
        for law in cases:
            # Calculate joint FP directly from the literal (0,1) state.
            fp = sum(law[i, 3, k] for i, k in itertools.product(range(4), repeat=2))
            limits = impl.law_limits(law)
            for eps in [fp, min(F(1), fp + F(1, 10))]:
                for mode in ['N', 'N6', 'F']:
                    got = impl.lp_interval(impl.word_problem(limits, eps, mode))
                    self.assertEqual(got['status'], 'ok')
                    np.testing.assert_allclose([got['lower'], got['upper']],
                                               oracle(law, eps, mode), atol=1e-8, rtol=0)

    def test_asymmetric_prevalence_analytic_truth_and_joint_fp(self):
        for m in [F(1, 5), F(4, 5)]:
            law = impl.markov_law(prevalence=m, rho=F(3, 5), phi=F(1, 10))
            self.assertEqual(sum(law.flat), 1)
            # X==1 is exactly states 1 and 2, independent of feature helpers.
            spin = [-1, 1, 1, -1]
            theta = sum(law[i,j,k]*spin[i]*spin[j]
                        for i,j,k in itertools.product(range(4), repeat=3))
            fp = sum(law[i,3,k] for i,k in itertools.product(range(4), repeat=2))
            self.assertEqual(theta, 1-4*m*(1-m)*(1-F(3,5)))
            self.assertEqual(fp, (1-m)*F(1,10))

    def test_exact_sparse_gluing_preserves_both_pairs_and_full_h(self):
        for law in [cyclic_fixture([[0], [2]]), cyclic_fixture([[0,1,3,2,2], [3,0]])]:
            k = law.sum(axis=2)
            h = np.zeros((2,4,2), dtype=object)
            y = [0,0,1,1]
            for i,z,j in itertools.product(range(4), repeat=3):
                h[y[i],z,y[j]] += law[i,z,j]
            t = impl.glue_triple(k,h)
            np.testing.assert_array_equal(t.sum(axis=0), k)
            np.testing.assert_array_equal(t.sum(axis=2), k)
            hh = np.zeros((2,4,2), dtype=object)
            for i,z,j in itertools.product(range(4), repeat=3):
                hh[y[i],z,y[j]] += t[i,z,j]
            np.testing.assert_array_equal(hh,h)
            self.assertTrue(all(impl.check_gluing(k,h,t,exact=True).values()))
            broken = t.copy(); broken[0,0,0] += F(1,1000)
            self.assertFalse(all(impl.check_gluing(k,h,broken,exact=True).values()))

    def test_exact_dual_certificate_rechecked_without_sympy_checker(self):
        law = cyclic_fixture([[0,1,2,3], [1,2,0]])
        problem = impl.word_problem(impl.law_limits(law), F(1), 'F')
        got = impl.lp_interval(problem, return_solutions=True)
        for sign, result in zip([1,-1], got['solutions']):
            cert = impl.exact_certificate(problem,result,sign)
            self.assertTrue(cert['verified'], cert)
            p = list(map(F, cert['primal']))
            de = list(map(F, cert['dual_equality']))
            du = list(map(F, cert['dual_inequality']))
            self.assertTrue(all(v>=0 for v in p))
            self.assertTrue(all(v<=0 for v in du))
            dot = lambda a,b: sum(F(str(x))*y for x,y in zip(a,b))
            for a,b in zip(problem.ae,problem.be): self.assertEqual(dot(a,p), F(str(b)))
            for a,b in zip(problem.au,problem.bu): self.assertLessEqual(dot(a,p), F(str(b)))
            for col,c in enumerate(problem.c):
                self.assertGreaterEqual(F(int(sign*c))-dot(problem.ae[:,col],de)-dot(problem.au[:,col],du),0)
            self.assertEqual(dot(sign*problem.c,p), dot(problem.be,de)+dot(problem.bu,du))


class IndependentInferenceTests(unittest.TestCase):
    def test_binomial_boundary_formulas_and_invalid_arguments(self):
        for n in [1,10,500]:
            self.assertAlmostEqual(inference.cp_upper(0,n,.01),1-.01**(1/n),places=12)
            self.assertEqual(inference.cp_upper(n,n,.01),1)
            self.assertAlmostEqual(inference.cp_interval(0,n,.05)[1],1-.025**(1/n),places=12)
        for args in [(0,0),(2,1),(-1,2),(1.5,10)]:
            with self.assertRaises(ValueError): inference.cp_upper(*args)

    def test_refutation_has_precedence_over_loss_exclusion(self):
        self.assertEqual(inference.coverage_decision([(.935,.944)]*9),'refutation_signal')
        self.assertEqual(inference.coverage_decision([(.935,.970)]*9),'large_loss_excluded')
        self.assertEqual(inference.coverage_decision([(.935,.970)]*8),'inconclusive')

    def test_shared_cell_box_is_nested(self):
        # Deterministic balanced cell counts, not draws from a scientific grid.
        y=np.asarray(list(itertools.product([0,1],repeat=3))*20)
        ay=np.asarray(list(itertools.product([0,1],repeat=3))*40)
        ax=np.repeat([0,1],160)
        got=inference.shared_box_comparison(y,ax,ay,1.)
        self.assertEqual(got['nesting_violations'],[])
        iv=got['intervals']
        for weak,strong in [('N','N6'),('N6','N8'),('N8','F')]:
            self.assertEqual(iv[weak]['status'],'ok')
            self.assertLessEqual(iv[weak]['lower'],iv[strong]['lower']+1e-8)
            self.assertGreaterEqual(iv[weak]['upper'],iv[strong]['upper']-1e-8)

    def test_oracle_argument_cannot_affect_learned_interval(self):
        y=np.asarray(list(itertools.product([0,1],repeat=3))*5)
        ax=np.array([0,1]*20)
        cx=np.array([0,1]*10); cy=np.array([0,0]*10)
        a=inference.infer_intervals(y,ax,y,cx,cy,0.)['learned_upper']
        b=inference.infer_intervals(y,ax,y,cx,cy,1.)['learned_upper']
        self.assertEqual(a,b)
        self.assertGreater(a['epsilon'],0.)

    def test_partial_grid_cannot_receive_a_confirmatory_verdict(self):
        rows=[]
        for cid in range(9):
            rows.append(dict(condition_id=cid,method='learned_upper',family='primary',seed=410001,
                status='ok',contains_population=True,covers_truth=True,fallback_contains_population=True,
                width=1.,width_minus_oracle=0.,empty_feasible=False,solver_failure=False,
                fallback_used=False,epsilon_covers_f=True,upper_minus_f=.01))
        result=inference.summarize(rows,bootstrap_reps=10,smoke=False)
        self.assertFalse(result['primary_grid_complete'])
        self.assertEqual(result['coverage_decision'],'incomplete_protocol')
        self.assertEqual(result['practical_precision'],'incomplete_protocol')


class IndependentAdapterTests(unittest.TestCase):
    def test_one_adjacent_gold_is_sufficient(self):
        rows,manifest=adapter.fixture()
        del rows[-1]['x2_gold']
        got=adapter.validate_dataset(rows,manifest)
        self.assertEqual(got['status'],'eligible_iid_validation',got)

    def test_unequal_inclusion_probabilities_need_different_inference(self):
        rows,manifest=adapter.fixture()
        another=copy.deepcopy(rows[1]);another.update(episode_id='new',unit_id='new',selection_probability=.1)
        rows.append(another)
        got=adapter.validate_dataset(rows,manifest)
        self.assertEqual(got['status'],'blocked_selection',got)
        self.assertFalse(got['coverage_enabled'])

    def test_overlapping_units_and_proxy_gold_are_not_certified(self):
        rows,manifest=adapter.fixture()
        rows[1]['unit_id']=rows[2]['unit_id']
        self.assertEqual(adapter.validate_dataset(rows,manifest)['status'],'dependent_inference_required')
        rows,manifest=adapter.fixture();rows[1]['gold_source']='classifier_proxy'
        self.assertEqual(adapter.validate_dataset(rows,manifest)['status'],'blocked_gold')

    def test_missing_self_weighting_evidence_is_not_an_iid_sample_certificate(self):
        rows,manifest=adapter.fixture()
        manifest.pop('self_weighting_sample',None)
        got=adapter.validate_dataset(rows,manifest)
        self.assertEqual(got['status'],'blocked_selection',got)


if __name__ == '__main__':
    unittest.main()
