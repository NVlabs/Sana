#!/usr/bin/env bash
set -euo pipefail

# This download and its SHA-256 pass are intentionally compute-node-only.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error: run this script inside a Slurm compute allocation; never on a login node" >&2
  echo "example: srun -A nvr_elm_llm -p cpu_short -N1 -n1 -c1 --mem=2G -t 00:10:00 bash $0" >&2
  exit 2
fi

readonly TAEHV_COMMIT="32ac0146b11007cda5a57b60a3b35653361fb8a4"
readonly TAEHV_SHA256="007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339"
readonly TAEHV_URL="https://raw.githubusercontent.com/madebyollin/taehv/${TAEHV_COMMIT}/taeltx2_3_wide.pth"
readonly TAEHV_DIR="/lustre/fsw/portfolios/nvr/users/yitongl/pretrained_models/LTX-2.5-public/taehv"
readonly TAEHV_PATH="${TAEHV_DIR}/taeltx2_3_wide.pth"
readonly PARTIAL_PATH="${TAEHV_PATH}.partial.${SLURM_JOB_ID}.${BASHPID}"

command -v curl >/dev/null || {
  echo "error: curl is required on the compute node" >&2
  exit 2
}
command -v sha256sum >/dev/null || {
  echo "error: sha256sum is required on the compute node" >&2
  exit 2
}

mkdir -p -- "${TAEHV_DIR}"

if [[ -f "${TAEHV_PATH}" ]]; then
  current_sha="$(sha256sum "${TAEHV_PATH}" | awk '{print $1}')"
  if [[ "${current_sha}" == "${TAEHV_SHA256}" ]]; then
    echo "TAEHV wide weight is already installed and verified: ${TAEHV_PATH}"
    exit 0
  fi
  echo "error: existing weight has unexpected SHA-256 and was not overwritten" >&2
  echo "path:     ${TAEHV_PATH}" >&2
  echo "expected: ${TAEHV_SHA256}" >&2
  echo "actual:   ${current_sha}" >&2
  exit 1
fi

cleanup() {
  rm -f -- "${PARTIAL_PATH}"
}
trap cleanup EXIT

curl \
  --fail \
  --location \
  --proto '=https' \
  --retry 3 \
  --retry-all-errors \
  --output "${PARTIAL_PATH}" \
  "${TAEHV_URL}"

download_sha="$(sha256sum "${PARTIAL_PATH}" | awk '{print $1}')"
if [[ "${download_sha}" != "${TAEHV_SHA256}" ]]; then
  echo "error: downloaded TAEHV weight failed SHA-256 verification" >&2
  echo "expected: ${TAEHV_SHA256}" >&2
  echo "actual:   ${download_sha}" >&2
  exit 1
fi

mv -- "${PARTIAL_PATH}" "${TAEHV_PATH}"
trap - EXIT
echo "Installed and verified TAEHV wide weight: ${TAEHV_PATH}"
