#!/usr/bin/env bash
set -euo pipefail

: "${OMP_NUM_THREADS:=4}" "${MKL_NUM_THREADS:=4}" "${OPENBLAS_NUM_THREADS:=4}"
: "${NUMEXPR_NUM_THREADS:=4}" "${TF_NUM_INTRAOP_THREADS:=1}" "${TF_NUM_INTEROP_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
export TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIPER_OPENVLA_LOG_QUEUE_SIZE="${PIPER_OPENVLA_LOG_QUEUE_SIZE:-1024}"
export PYTHONNOUSERSITE=1
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

if (( $# < 1 )); then
  printf '%s\n' 'usage: run_piper_openvla_runtime.sh {sync|async} [options]' >&2
  exit 2
fi

runtime_kind="$1"
shift
if [[ "${runtime_kind}" != "sync" && "${runtime_kind}" != "async" ]]; then
  printf 'error=invalid_runtime_kind value=%q\n' "${runtime_kind}" >&2
  exit 2
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
installed_prefix=""
if [[ -r "${project_root}/.install-prefix" ]]; then
  IFS= read -r installed_prefix <"${project_root}/.install-prefix"
fi
installed_lerobot_prefix=""
if [[ -r "${project_root}/.lerobot-prefix" ]]; then
  IFS= read -r installed_lerobot_prefix <"${project_root}/.lerobot-prefix"
fi
installed_piper_repo=""
if [[ -r "${project_root}/.piper-repo" ]]; then
  IFS= read -r installed_piper_repo <"${project_root}/.piper-repo"
fi
openvla_prefix="${OPENVLA_CONDA_PREFIX:-${installed_prefix}}"
robot_prefix="${LEROBOT_CONDA_PREFIX:-${PIPER_CONDA_PREFIX:-${installed_lerobot_prefix}}}"
[[ -n "${openvla_prefix}" ]] || {
  printf '%s\n' 'error=runtime_environment_missing hint=run_scripts/install_rtx4090.sh' >&2
  exit 1
}
if [[ -z "${PIPER_REPO:-}" && -n "${installed_piper_repo}" ]]; then
  export PIPER_REPO="${installed_piper_repo}"
fi
openvla_python="${openvla_prefix}/bin/python"
robot_python="${robot_prefix}/bin/python"
mkdir -p "${project_root}/artifacts"
server_log="${project_root}/artifacts/piper_openvla_topic_server.log"
forward_args=()
checkpoint_override=""
action_chunk_override=""
base_model_override=""
oft_repo_override=""
lerobot_prefix_override=""
while (( $# > 0 )); do
  case "$1" in
    --checkpoint)
      if (( $# < 2 )); then
        printf '%s\n' 'error=--checkpoint requires a path' >&2
        exit 2
      fi
      if [[ "$2" == hf://* ]]; then checkpoint_override="$2"; else checkpoint_override="$(readlink -m "$2")"; fi
      shift 2
      ;;
    --checkpoint=*)
      checkpoint_value="${1#--checkpoint=}"
      if [[ "${checkpoint_value}" == hf://* ]]; then checkpoint_override="${checkpoint_value}"; else checkpoint_override="$(readlink -m "${checkpoint_value}")"; fi
      shift
      ;;
    --base-model)
      (( $# >= 2 )) || { printf '%s\n' 'error=--base-model requires a source' >&2; exit 2; }
      base_model_override="$2"
      shift 2
      ;;
    --base-model=*)
      base_model_override="${1#--base-model=}"
      shift
      ;;
    --openvla-oft-repo)
      (( $# >= 2 )) || { printf '%s\n' 'error=--openvla-oft-repo requires a path' >&2; exit 2; }
      oft_repo_override="$(readlink -m "$2")"
      shift 2
      ;;
    --openvla-oft-repo=*)
      oft_repo_override="$(readlink -m "${1#--openvla-oft-repo=}")"
      shift
      ;;
    --lerobot-prefix)
      (( $# >= 2 )) || { printf '%s\n' 'error=--lerobot-prefix requires a path' >&2; exit 2; }
      lerobot_prefix_override="$(readlink -m "$2")"
      shift 2
      ;;
    --lerobot-prefix=*)
      lerobot_prefix_override="$(readlink -m "${1#--lerobot-prefix=}")"
      shift
      ;;
    --action-chunk)
      if (( $# < 2 )); then
        printf '%s\n' 'error=--action-chunk requires a positive integer' >&2
        exit 2
      fi
      action_chunk_override="$2"
      shift 2
      ;;
    --action-chunk=*)
      action_chunk_override="${1#--action-chunk=}"
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

launch_environment=()
if [[ -n "${checkpoint_override}" ]]; then
  launch_environment+=("PIPER_OPENVLA_CHECKPOINT=${checkpoint_override}")
fi
if [[ -n "${action_chunk_override}" ]]; then
  launch_environment+=("PIPER_ACTION_CHUNK=${action_chunk_override}")
fi

test -x "${openvla_python}"
cd "${project_root}"

readarray -t launch_values < <(
  env "${launch_environment[@]}" \
    "${openvla_python}" -m openvla_pipeline.launch_plan \
    "${runtime_kind}" --lines "${forward_args[@]}"
)
if (( ${#launch_values[@]} != 9 )); then
  printf 'error=invalid_launch_plan_fields expected=9 actual=%s\n' \
    "${#launch_values[@]}" >&2
  exit 2
fi

config_source="${launch_values[0]}"
server_url="${launch_values[1]}"
checkpoint="${launch_values[2]}"
server_host="${launch_values[3]}"
server_port="${launch_values[4]}"
auto_start_local_server="${launch_values[5]}"
action_chunk="${launch_values[6]}"
auth_token_env="${launch_values[7]}"
health_attempts="${launch_values[8]}"

if [[ -n "${checkpoint}" ]]; then
  export PIPER_OPENVLA_CHECKPOINT="${checkpoint}"
fi
if [[ -n "${action_chunk}" ]]; then
  export PIPER_ACTION_CHUNK="${action_chunk}"
fi
if [[ -n "${base_model_override}" ]]; then
  export OPENVLA_BASE_MODEL="${base_model_override}"
fi
if [[ -n "${oft_repo_override}" ]]; then
  export OPENVLA_OFT_REPO="${oft_repo_override}"
fi
if [[ -n "${lerobot_prefix_override}" ]]; then
  robot_prefix="${lerobot_prefix_override}"
  robot_python="${robot_prefix}/bin/python"
fi
[[ -n "${robot_prefix}" ]] || {
  printf '%s\n' 'error=lerobot_environment_missing use=--lerobot-prefix_or_LEROBOT_CONDA_PREFIX' >&2
  exit 1
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'dry_run=True mode=%s runtime_config=%s server=%s checkpoint=%s action_chunk=%s auth_token_env=%s auto_start_local_server=%s\n' \
    "${runtime_kind}" \
    "${config_source}" \
    "${server_url}" \
    "${checkpoint:-unset}" \
    "${action_chunk:-metadata-unavailable}" \
    "${auth_token_env}" \
    "${auto_start_local_server}"
  exit 0
fi

test -x "${robot_python}"

server_started_here=0
server_pid=""
health_json=""
if health_json="$(curl --silent --fail --max-time 2 "${server_url}/health")"; then
  if [[ -n "${checkpoint}" ]]; then
    running_checkpoint="$(printf '%s' "${health_json}" | "${openvla_python}" -c 'import json,sys; h=json.load(sys.stdin); print(h.get("checkpoint_source", h["checkpoint"]))')"
    if [[ "${checkpoint}" == hf://* ]]; then
      checkpoint_matches=$([[ "${running_checkpoint}" == "${checkpoint}" ]] && printf 1 || printf 0)
    else
      checkpoint_matches=$([[ "$(realpath -m "${running_checkpoint}")" == "$(realpath -m "${checkpoint}")" ]] && printf 1 || printf 0)
    fi
    if [[ "${checkpoint_matches}" != "1" ]]; then
      printf 'OpenVLA server checkpoint mismatch: running=%s expected=%s\n' \
        "${running_checkpoint}" "${checkpoint}" >&2
      exit 1
    fi
  fi
else
  if [[ "${auto_start_local_server}" != "1" ]]; then
    printf 'Remote OpenVLA server is unavailable; local auto-start is disabled: %s\n' \
      "${server_url}" >&2
    exit 2
  fi
  if [[ -z "${checkpoint}" ]]; then
    printf '%s\n' \
      'No model server is running and no checkpoint is configured. Set PIPER_OPENVLA_CHECKPOINT.' >&2
    exit 2
  fi
  if [[ "${checkpoint}" != hf://* && ! -f "${checkpoint}/checkpoint_metadata.json" ]]; then
    printf 'OpenVLA checkpoint is incomplete: %s\n' "${checkpoint}" >&2
    exit 1
  fi

  PIPER_OPENVLA_CHECKPOINT="${checkpoint}" \
    OPENVLA_BASE_MODEL="${base_model_override:-${OPENVLA_BASE_MODEL:-}}" \
    OPENVLA_OFT_REPO="${oft_repo_override:-${OPENVLA_OFT_REPO:-}}" \
    "${project_root}/tools/run_piper_openvla_server.sh" \
      --config "${config_source}" --host "${server_host}" --port "${server_port}" \
      >>"${server_log}" 2>&1 &
  server_pid=$!
  server_started_here=1

  for _ in $(seq 1 "${health_attempts}"); do
    if curl --silent --fail --max-time 2 "${server_url}/health" >/dev/null; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      printf 'OpenVLA server failed. log=%s\n' "${server_log}" >&2
      exit 1
    fi
    sleep 1
  done
  if ! curl --silent --fail --max-time 2 "${server_url}/health" >/dev/null; then
    printf 'OpenVLA server health timeout after %s seconds. log=%s\n' \
      "${health_attempts}" "${server_log}" >&2
    exit 1
  fi
fi

cleanup() {
  if (( server_started_here == 1 )) && [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${runtime_kind}" == "sync" ]]; then
  runtime_module="openvla_pipeline"
else
  runtime_module="openvla_async_pipeline"
fi
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  "${robot_python}" -m "${runtime_module}" deploy "${forward_args[@]}"
