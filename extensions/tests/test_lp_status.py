"""LP status handling: infeasibility is accepted only after a presolve-off retry, an empty interval carries no endpoints, and
the occupancy sensitivity script keeps empty sets and solver failures apart."""
import importlib.util, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
from c11_upgrade.population import lp_interval

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/occupancy_sensitivity.py'

class EmptyAndRetry(unittest.TestCase):
    def test_infeasible_problem_is_empty_without_endpoints(self):
        # x0 + x1 = 1 and x0 + x1 <= 0.5 with x >= 0: infeasible under both presolve settings.
        p = SimpleNamespace(c=np.array([1., -1.]), au=[[1., 1.]], bu=[.5], ae=[[1., 1.]], be=[1.])
        r = lp_interval(p)
        self.assertEqual(r['status'], 'empty'); self.assertIsNone(r['lower']); self.assertIsNone(r['upper'])

    def test_feasible_problem_unchanged(self):
        p = SimpleNamespace(c=np.array([1., -1.]), au=[[1., 0.]], bu=[.75], ae=[[1., 1.]], be=[1.])
        r = lp_interval(p)
        self.assertEqual(r['status'], 'ok'); self.assertAlmostEqual(r['lower'], -1.); self.assertAlmostEqual(r['upper'], .5)

class SensitivityStatuses(unittest.TestCase):
    """occupancy_sensitivity.finite_sample: a certified empty set is the flagged fallback [-1, 1] and contains the target; a solver
    failure is counted separately and never contains the target; an unknown status is rejected."""
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('occupancy_sensitivity', SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.mod)
        cls.ws = [(None, [(1, 0), (1, 0), (1, 0)])] * 4

    def run_with(self, status):
        out = {'status': status, 'lower': None, 'upper': None}
        with mock.patch.object(self.mod, 'solve_information', lambda *a: {k: dict(out) for k in 'MNF'}):
            return self.mod.finite_sample(self.ws, 1.0, 1, 1)

    def test_empty_set_is_flagged_fallback(self):
        for v in self.run_with('empty').values():
            self.assertEqual((v['empty_sets'], v['solver_failures'], v['contains_record_target'], v['mean_width']), (1, 0, 1.0, 2.0))

    def test_solver_failure_is_counted_separately_without_coverage(self):
        for v in self.run_with('solver_failure').values():
            self.assertEqual((v['empty_sets'], v['solver_failures'], v['contains_record_target'], v['mean_width']), (0, 1, 0.0, 2.0))

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError): self.run_with('iteration_limit')

if __name__ == '__main__':
    unittest.main()
