#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
runtime_prefix="${OPENVLA_CONDA_PREFIX:-}"
if [[ -z "${runtime_prefix}" && -r "${project_root}/.install-prefix" ]]; then
  IFS= read -r runtime_prefix <"${project_root}/.install-prefix"
fi
[[ -n "${runtime_prefix}" && -x "${runtime_prefix}/bin/python" ]] || {
  printf '%s\n' 'error=openvla_runtime_missing hint=run_scripts/install_rtx4090.sh' >&2
  exit 1
}
cd "${project_root}"
exec env PYTHONNOUSERSITE=1 "${runtime_prefix}/bin/python" \
  scripts/train_openvla_lora.py "$@"
