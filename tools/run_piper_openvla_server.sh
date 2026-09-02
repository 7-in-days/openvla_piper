#!/usr/bin/env bash
set -euo pipefail

: "${OMP_NUM_THREADS:=4}" "${MKL_NUM_THREADS:=4}" "${OPENBLAS_NUM_THREADS:=4}"
: "${NUMEXPR_NUM_THREADS:=4}" "${TF_NUM_INTRAOP_THREADS:=1}" "${TF_NUM_INTEROP_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
export TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIPER_OPENVLA_LOG_QUEUE_SIZE="${PIPER_OPENVLA_LOG_QUEUE_SIZE:-1024}"
for bounded_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  bounded_value="${!bounded_name}"
  [[ "${bounded_value}" =~ ^[1-9][0-9]*$ ]] && (( bounded_value <= 8 )) || {
    printf 'error=invalid_env_limit name=%s value=%q allowed=1..8\n' "${bounded_name}" "${bounded_value}" >&2
    exit 2
  }
done
for bounded_name in TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS; do
  bounded_value="${!bounded_name}"
  [[ "${bounded_value}" =~ ^[1-9][0-9]*$ ]] && (( bounded_value <= 4 )) || {
    printf 'error=invalid_env_limit name=%s value=%q allowed=1..4\n' "${bounded_name}" "${bounded_value}" >&2
    exit 2
  }
done

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
installed_prefix=""
if [[ -r "${project_root}/.install-prefix" ]]; then
  IFS= read -r installed_prefix <"${project_root}/.install-prefix"
fi
installed_oft_repo=""
if [[ -r "${project_root}/.openvla-oft-repo" ]]; then
  IFS= read -r installed_oft_repo <"${project_root}/.openvla-oft-repo"
fi
if [[ -z "${OPENVLA_OFT_REPO:-}" && -n "${installed_oft_repo}" ]]; then
  export OPENVLA_OFT_REPO="${installed_oft_repo}"
fi
openvla_prefix="${OPENVLA_CONDA_PREFIX:-${installed_prefix}}"
[[ -n "${openvla_prefix}" ]] || {
  printf '%s\n' 'error=runtime_environment_missing hint=run_scripts/install_rtx4090.sh' >&2
  exit 1
}
python_bin="${openvla_prefix}/bin/python"
runtime_config="${PIPER_OPENVLA_RUNTIME_CONFIG:-}"
config_from_cli=0
forward_args=()
while (( $# > 0 )); do
  case "$1" in
    --config)
      if (( $# < 2 )); then
        printf '%s\n' 'error=--config requires a path' >&2
        exit 2
      fi
      runtime_config="$(readlink -f "$2")"
      test -f "${runtime_config}"
      forward_args+=(--config "${runtime_config}")
      config_from_cli=1
      shift 2
      ;;
    --config=*)
      runtime_config="$(readlink -f "${1#--config=}")"
      test -f "${runtime_config}"
      forward_args+=(--config "${runtime_config}")
      config_from_cli=1
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done
config_args=()
config_source="${project_root}/openvla_pipeline/user_settings.py"
if [[ -n "${runtime_config}" ]]; then
  test -f "${runtime_config}"
  runtime_config="$(readlink -f "${runtime_config}")"
  if (( config_from_cli == 0 )); then
    config_args=(--config "${runtime_config}")
  fi
  config_source="${runtime_config}"
fi

test -x "${python_bin}"
mkdir -p "${project_root}/artifacts/huggingface"

"${project_root}/scripts/preflight_rtx4090.sh" --skip-ros-check

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'dry_run=True runtime_config=%s\n' "${config_source}"
  printf 'command=%q -m openvla_pipeline serve' "${python_bin}"
  if (( ${#config_args[@]} > 0 )); then
    printf ' %q' "${config_args[@]}"
  fi
  if (( ${#forward_args[@]} > 0 )); then
    printf ' %q' "${forward_args[@]}"
  fi
  printf '\n'
  exit 0
fi

cd "${project_root}"
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export ROBOT_PLATFORM=PIPER
export HF_HOME="${HF_HOME:-${project_root}/artifacts/huggingface}"
export TF_CPP_MIN_LOG_LEVEL=3

exec "${python_bin}" -m openvla_pipeline serve "${config_args[@]}" "${forward_args[@]}"
