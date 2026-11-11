"""Tests for repository-local output paths and relative execution-source hashes in run plans."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import runtime as r

class Portability(unittest.TestCase):
 def test_generated_runs_allowed_but_archived_outputs_protected(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);(repo/'configs').mkdir()
   c=dict(stage='smoke',topic='C11',seeds=[1],episodes=40,stay=[.5],audit_budgets=[10],lengths=[3],moment_sets=['marginal','neighbor'],intervals=['hoeffding'])
   (repo/'configs/smoke.json').write_text(json.dumps(c))
   protected=repo/'results/runs/must-not-create'
   with self.assertRaisesRegex(ValueError,'archived results'):r.execute(c,protected,repo=repo)
   self.assertFalse(protected.exists())
   before=r.source_hashes(repo);out=repo/'results/generated/run'
   m=r.execute(c,out,repo=repo)
   self.assertEqual(m['status'],'PASS');r.verify_outputs(out)
   self.assertEqual(before,r.source_hashes(repo))
 def test_relative_execution_source_is_resolved_against_repo(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td)/'repo';repo.mkdir();(repo/'run.py').write_text('source')
   c=dict(job_id='C11-portability-test');plan=Path(td)/'plan.json'
   r.atomic_json(plan,dict(execution_sources={'run.py':r.sha(repo/'run.py')},jobs=[dict(id=c['job_id'],config_hash=r.object_hash(c),source_hashes=r.source_hashes(repo))]))
   r.check_plan(c,plan,repo=repo)
   (repo/'run.py').write_text('changed')
   with self.assertRaisesRegex(ValueError,'Execution source changed'):r.check_plan(c,plan,repo=repo)
if __name__=='__main__':unittest.main()
