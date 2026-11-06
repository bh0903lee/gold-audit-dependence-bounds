"""Download the 81 SPHERE house source files read by extensions/scripts/sphere_pir.py from the University of Bristol data
repository (doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p, version of 2022-11-04) and check them against the SHA-256 digests
stored in manuscript/supplement/extensions/sphere_summary.json. The files are not redistributed with this package; their
terms of use are those published with the dataset (data/raw/sphere_house/DATA_SOURCE.md).

Standard library only. This is the only command of the package that uses the network. A file that is already present is
kept if its digest matches and reported (never overwritten) if it does not; a downloaded file is kept only if its digest
matches. Exit code 0 if all 81 files are present with the stored digests, 1 otherwise.

  python -I extensions/scripts/fetch_sphere.py [OUT_DIR]   # default: data/raw/sphere_house/raw
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://data.bris.ac.uk/datasets/2h0wyctxrd69j2oqccsi45hy1p/data/raw/'


def sha256(data): return hashlib.sha256(data).hexdigest()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data/raw/sphere_house/raw'
    digests = json.loads((ROOT / 'manuscript/supplement/extensions/sphere_summary.json').read_text())['input_sha256']
    bad = 0
    for name, digest in sorted(digests.items()):
        path = out / name
        if path.is_file():
            if sha256(path.read_bytes()) != digest:
                print(f'MISMATCH   {name} (existing file kept; remove it to download again)'); bad += 1
            continue
        with urllib.request.urlopen(BASE + name, timeout=60) as r: data = r.read()
        if sha256(data) != digest:
            print(f'MISMATCH   {name} (downloaded file discarded)'); bad += 1; continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part'); tmp.write_bytes(data); tmp.rename(path)
        print(f'ok         {name}')
    print(f'{len(digests) - bad} of {len(digests)} files present with the stored digests in {out}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
