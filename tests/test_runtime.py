import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import runtime as r

class Runtime(unittest.TestCase):
    def test_stage_seeds_are_disjoint(self):
        sets=[set(json.loads((ROOT/'configs'/f'{s}.json').read_text())['seeds']) for s in ['smoke','poc','main']]
        self.assertTrue(all(not sets[i]&sets[j] for i in range(3) for j in range(i)))

    def test_frozen_scientific_parameters(self):
        config=json.loads((ROOT/'configs/smoke.json').read_text()); r.validate_config(config)
        changed=dict(config,unreviewed_parameter=1)
        with self.assertRaisesRegex(ValueError,'differs from the canonical'):r.validate_config(changed)
        changed=dict(config,seeds=[-1])
        with self.assertRaisesRegex(ValueError,'seed'):r.validate_config(changed)

    def test_main_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'forbidden'
            args=[sys.executable,str(ROOT/'run.py'),'--stage','main','--out',str(out)]
            for extra in [[],['--execute-main']]:
                result=subprocess.run(args+extra,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Main execution disabled',result.stderr)
                self.assertFalse(out.exists())

    def test_config_stage_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--stage','smoke','--config',str(ROOT/'configs/main.json'),'--out',str(Path(td)/'forbidden')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((Path(td)/'forbidden').exists())

    def test_atomic_json_preserves_valid_file_on_nan(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.json';r.atomic_json(path,{'valid':1})
            with self.assertRaises(ValueError):r.atomic_json(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'valid':1})
            self.assertEqual(list(Path(td).glob('*.tmp')),[])

    def test_artifact_corruption_detected(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);r.atomic_json(out/'rows.json',[{'x':1}]);r.atomic_json(out/'analysis.json',{})
            hashes={name:r.sha(out/name) for name in ['rows.json','analysis.json']}
            r.atomic_json(out/'manifest.json',{'status':'PASS','output_hashes':hashes})
            r.verify_outputs(out);(out/'rows.json').write_text('[]')
            with self.assertRaises(ValueError):r.verify_outputs(out)

    def test_main_requires_matching_plan(self):
        # Exercises only the main-stage plan check; the engine is not run.
        config=json.loads((ROOT/'configs/main.json').read_text());config['job_id']='test-plan-only'
        with tempfile.TemporaryDirectory() as td:
            plan=Path(td)/'plan.json'
            job=dict(id=config['job_id'],config_hash=r.object_hash(config),source_hashes=r.source_hashes())
            r.atomic_json(plan,{'jobs':[job]});r.check_plan(config,plan)
            bad=copy.deepcopy(config);bad['seeds']=[-1]
            with self.assertRaises(ValueError):r.check_plan(bad,plan)
            r.atomic_json(plan,{'jobs':[dict(job,source_hashes={})]})
            with self.assertRaisesRegex(ValueError,'Repository files changed'):r.check_plan(config,plan)
            with self.assertRaisesRegex(ValueError,'Main execution disabled'):r.check_plan(config,None)

    def test_plan_is_written_outside_hashed_files(self):
        # A plan in a hashed location would change the hashes it binds; plan_main.py refuses it before writing.
        target=ROOT/'plan_location_test.json'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/plan_main.py'),'--out',str(target),'--seeds','91001'],capture_output=True,text=True,cwd=ROOT)
        self.assertNotEqual(result.returncode,0);self.assertFalse(target.exists());self.assertIn('results/',result.stderr)
        self.assertFalse(any(k.startswith('results/') for k in r.source_hashes()))

    def test_input_checksum_required(self):
        config=json.loads((ROOT/'configs/smoke.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'input';file.write_text('test')
            config.update(data=str(file),data_sha256='bad')
            with self.assertRaisesRegex(ValueError,'checksum'):r.validate_config(config)

    def test_repository_manifest_rejects_added_changed_missing_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);entry=repo/'run.py';entry.write_text('original')
            def freeze():
                r.atomic_json(repo/'repository_manifest.json',{'files':[{'path':k,'sha256':v} for k,v in r.source_hashes(repo).items()]})
            freeze();self.assertEqual(r.verify_repository(repo)['files'],1)
            entry.write_text('changed')
            with self.assertRaises(ValueError):r.verify_repository(repo)
            freeze();added=repo/'unexpected.py';added.write_text('extra')
            with self.assertRaises(ValueError):r.verify_repository(repo)
            freeze();added.unlink()
            with self.assertRaises(ValueError):r.verify_repository(repo)
            freeze();m=json.loads((repo/'repository_manifest.json').read_text());m['files']*=2;r.atomic_json(repo/'repository_manifest.json',m)
            with self.assertRaises(ValueError):r.verify_repository(repo)

if __name__=='__main__':unittest.main(verbosity=2)
