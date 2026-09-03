#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
lerobot_prefix="${LEROBOT_CONDA_PREFIX:-}"
rlds_prefix="${RLDS_TOOLS_PREFIX:-${project_root}/.runtime/envs/rlds-tools}"

if [[ -z "${lerobot_prefix}" && -r "${project_root}/.lerobot-prefix" ]]; then
  IFS= read -r lerobot_prefix <"${project_root}/.lerobot-prefix"
fi
if [[ -z "${lerobot_prefix}" || ! -x "${lerobot_prefix}/bin/python" ]]; then
  printf '%s\n' 'error=lerobot_environment_missing hint=run_scripts/install_openvla.sh_first' >&2
  exit 1
fi

if [[ ! -x "${rlds_prefix}/bin/python" ]]; then
  mkdir -p "$(dirname "${rlds_prefix}")"
  "${lerobot_prefix}/bin/python" -m venv --system-site-packages "${rlds_prefix}"
fi

"${rlds_prefix}/bin/python" -m pip install --disable-pip-version-check \
  -r "${project_root}/requirements-rlds.txt"

printf '%s\n' "${rlds_prefix}" >"${project_root}/.rlds-tools-prefix"
env PYTHONNOUSERSITE=1 \
  HF_HOME="${project_root}/.runtime/huggingface-rlds" \
  HF_DATASETS_CACHE="${project_root}/.runtime/huggingface-rlds/datasets" \
  "${rlds_prefix}/bin/python" -c \
  'import importlib.metadata as m; import lerobot, tensorflow as tf, tensorflow_datasets; print("rlds_tools=ready tensorflow={} tfds={}".format(tf.__version__, m.version("tensorflow-datasets")))'
