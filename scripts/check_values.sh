#!/bin/sh
# Value check in a temporary copy of the package: rerun the certificate checks, then scripts/recompute_values.py (integrity of
# the stored files, recomputed value files, every reported number); exit nonzero if any check fails.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd); tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp -R "$here/." "$tmp/pkg"; cd "$tmp/pkg"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
py=${PYTHON:-python}
echo '== stored four-point certificates (exact rational arithmetic) =='
$py -I -B manuscript/supplement/check_stored_certificates.py
echo '== appendix check and rebuild of the four certificates; gap and gluing certificates =='
$py -I -B manuscript/supplement/verify_theorem.py >/dev/null
$py -I -B manuscript/supplement/extensions/verify_fp_certificates.py >/dev/null
for f in verify_theorem.json extensions/fp_certificate_verification.json; do
  cmp "$here/manuscript/supplement/$f" "manuscript/supplement/$f"
  echo "identical  $f"
done
echo '== stored files, recomputed value files and reported numbers =='
$py -I -B scripts/recompute_values.py
echo 'Value check passed.'
