#!/usr/bin/env bash
# Identity-free one-node 1/2/4-GPU diagnostic. Supply account and partition to
# sbatch; this file deliberately contains neither.
#SBATCH --job-name=nanochat-gpu-preflight
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=120G
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${PREFLIGHT_MANIFEST:?Set PREFLIGHT_MANIFEST to the immutable JSON manifest}"
: "${PREFLIGHT_OUTPUT_ROOT:?Set PREFLIGHT_OUTPUT_ROOT to a new result directory}"
: "${NANOCHAT_CHECKOUT:?Set NANOCHAT_CHECKOUT to the clean bound checkout}"
for required_absolute in PREFLIGHT_MANIFEST PREFLIGHT_OUTPUT_ROOT NANOCHAT_CHECKOUT; do
  case "${!required_absolute}" in
    /*) ;;
    *) echo "${required_absolute} must be an absolute path" >&2; exit 2 ;;
  esac
done

MANIFEST_READER_PYTHON="${PREFLIGHT_MANIFEST_PYTHON:-python3}"
manifest_bootstrap_file="$(
  "${MANIFEST_READER_PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime_bootstrap"]["file"])' \
    "${PREFLIGHT_MANIFEST}"
)"
manifest_bootstrap_sha256="$(
  "${MANIFEST_READER_PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime_bootstrap"]["sha256"])' \
    "${PREFLIGHT_MANIFEST}"
)"
if [ "${manifest_bootstrap_file}" != "runtime_bootstrap.sh" ]; then
  echo "Manifest runtime bootstrap filename is not the fixed copied artifact" >&2
  exit 2
fi
if [ -n "${RUNTIME_BOOTSTRAP_SHA256:-}" ] && \
   [ "${RUNTIME_BOOTSTRAP_SHA256}" != "${manifest_bootstrap_sha256}" ]; then
  echo "RUNTIME_BOOTSTRAP_SHA256 differs from the manifest" >&2
  exit 2
fi
manifest_dir="$(cd "$(dirname "${PREFLIGHT_MANIFEST}")" && pwd)"
export RUNTIME_BOOTSTRAP="${RUNTIME_BOOTSTRAP:-${manifest_dir}/${manifest_bootstrap_file}}"
export RUNTIME_BOOTSTRAP_SHA256="${manifest_bootstrap_sha256}"
case "${RUNTIME_BOOTSTRAP}" in
  /*) ;;
  *) echo "RUNTIME_BOOTSTRAP must be an absolute path" >&2; exit 2 ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
  read -r observed_bootstrap_sha _ < <(sha256sum -- "${RUNTIME_BOOTSTRAP}")
elif command -v shasum >/dev/null 2>&1; then
  read -r observed_bootstrap_sha _ < <(shasum -a 256 -- "${RUNTIME_BOOTSTRAP}")
else
  echo "No SHA-256 utility is available to verify RUNTIME_BOOTSTRAP" >&2
  exit 2
fi
if [ "${observed_bootstrap_sha}" != "${RUNTIME_BOOTSTRAP_SHA256}" ]; then
  echo "RUNTIME_BOOTSTRAP hash mismatch; refusing to source it" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${RUNTIME_BOOTSTRAP}"

PYTHON_BIN="${NANOCHAT_PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "${NANOCHAT_CHECKOUT}"

exec "${PYTHON_BIN}" -m csc_fast_scaling_law.gpu_preflight orchestrate \
  --manifest "${PREFLIGHT_MANIFEST}" \
  --output-root "${PREFLIGHT_OUTPUT_ROOT}"
