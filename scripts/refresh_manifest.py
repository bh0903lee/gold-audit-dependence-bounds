"""Rewrite repository_manifest.json from the current repository files and manuscript/supplement/supplement_manifest.json from the
current stored results; run it after editing any hashed file."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import runtime as r
SUP=ROOT/'manuscript'/'supplement'
if __name__=='__main__':
 r.atomic_json(ROOT/'repository_manifest.json',dict(version=2,topic='C11',files=[dict(path=k,sha256=v) for k,v in r.source_hashes().items()],excluded=['results/','data/raw/','caches and build outputs','stored numerical results (JSON, CSV) in manuscript/supplement/ (listed in manuscript/supplement/supplement_manifest.json)','repository_manifest.json']))
 files=sorted(p.relative_to(SUP).as_posix() for p in SUP.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='supplement_manifest.json')
 r.atomic_json(SUP/'supplement_manifest.json',dict(description='SHA-256 digests of every file in manuscript/supplement/ except this one; checked by scripts/recompute_values.py',files=[dict(path=n,sha256=r.sha(SUP/n)) for n in files]))
 print(json.dumps(r.verify_repository()),len(files),'supplement files')
