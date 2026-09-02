#!/usr/bin/env bash
set -euo pipefail

skip_ros_check=0
warn_free_vram_mib="${OPENVLA_WARN_FREE_VRAM_MIB:-20000}"
require_free_vram_mib="${OPENVLA_REQUIRE_FREE_VRAM_MIB:-0}"
minimum_driver="${OPENVLA_MIN_NVIDIA_DRIVER:-570.26}"

usage() {
  cat <<'EOF'
usage: scripts/preflight_rtx4090.sh [options]

options:
  --skip-ros-check             do not require /opt/ros/humble/setup.bash
  --warn-free-vram-mib N       warning threshold only (default: 20000)
  --require-free-vram-mib N    fail below N MiB; default 0 means no install gate
  --help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --skip-ros-check)
      skip_ros_check=1
      shift
      ;;
    --warn-free-vram-mib)
      (( $# >= 2 )) || { printf '%s\n' 'error=--warn-free-vram-mib requires a value' >&2; exit 2; }
      warn_free_vram_mib="$2"
      shift 2
      ;;
    --require-free-vram-mib)
      (( $# >= 2 )) || { printf '%s\n' 'error=--require-free-vram-mib requires a value' >&2; exit 2; }
      require_free_vram_mib="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'error=unknown_option value=%q\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for numeric_value in "${warn_free_vram_mib}" "${require_free_vram_mib}"; do
  [[ "${numeric_value}" =~ ^[0-9]+$ ]] || {
    printf 'error=invalid_vram_threshold value=%q\n' "${numeric_value}" >&2
    exit 2
  }
done

os_release_file="${OPENVLA_OS_RELEASE_FILE:-/etc/os-release}"
[[ -r "${os_release_file}" ]] || {
  printf 'error=os_release_unreadable path=%q\n' "${os_release_file}" >&2
  exit 1
}
os_id="$(. "${os_release_file}"; printf '%s' "${ID:-}")"
os_version="$(. "${os_release_file}"; printf '%s' "${VERSION_ID:-}")"
if [[ "${os_id}" != "ubuntu" || ( "${os_version}" != "22.04" && "${os_version}" != "24.04" ) ]]; then
  printf 'error=unsupported_os id=%q version=%q expected=Ubuntu_22.04_or_24.04\n' \
    "${os_id}" "${os_version}" >&2
  exit 1
fi

nvidia_smi="${NVIDIA_SMI_BIN:-}"
if [[ -z "${nvidia_smi}" ]]; then
  nvidia_smi="$(command -v nvidia-smi || true)"
fi
[[ -n "${nvidia_smi}" && -x "${nvidia_smi}" ]] || {
  printf '%s\n' 'error=nvidia_driver_missing command=nvidia-smi' >&2
  exit 1
}

gpu_rows="$(${nvidia_smi} --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null)" || {
  printf '%s\n' 'error=nvidia_driver_unavailable nvidia-smi_query_failed=True' >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

gpu_name=""
driver_version=""
compute_capability=""
memory_total_mib=""
memory_free_mib=""
while IFS=',' read -r row_name row_driver row_cap row_total row_free; do
  row_name="$(trim "${row_name}")"
  if [[ "${row_name}" == *"RTX 4090"* ]]; then
    gpu_name="${row_name}"
    driver_version="$(trim "${row_driver}")"
    compute_capability="$(trim "${row_cap}")"
    memory_total_mib="$(trim "${row_total}")"
    memory_free_mib="$(trim "${row_free}")"
    break
  fi
done <<<"${gpu_rows}"

[[ -n "${gpu_name}" ]] || {
  printf '%s\n' 'error=target_gpu_missing expected=RTX_4090' >&2
  exit 1
}
[[ "${compute_capability}" == "8.9" ]] || {
  printf 'error=compute_capability_mismatch actual=%q expected=8.9\n' "${compute_capability}" >&2
  exit 1
}
[[ "${memory_total_mib}" =~ ^[0-9]+$ ]] || {
  printf 'error=invalid_total_vram value=%q\n' "${memory_total_mib}" >&2
  exit 1
}
if (( memory_total_mib < 23000 || memory_total_mib > 26000 )); then
  printf 'error=vram_mismatch actual_mib=%s expected_approx_mib=24564\n' "${memory_total_mib}" >&2
  exit 1
fi
if [[ "$(printf '%s\n%s\n' "${minimum_driver}" "${driver_version}" | sort -V | head -n1)" != "${minimum_driver}" ]]; then
  printf 'error=nvidia_driver_too_old actual=%q minimum=%q cuda_wheel=cu128\n' \
    "${driver_version}" "${minimum_driver}" >&2
  exit 1
fi

if (( skip_ros_check == 0 )); then
  ros_setup="${ROS_HUMBLE_SETUP:-/opt/ros/humble/setup.bash}"
  [[ -r "${ros_setup}" ]] || {
    printf 'error=ros2_humble_missing expected=%q hint=use_--skip-ros-check_only_if_ros_is_external\n' \
      "${ros_setup}" >&2
    exit 1
  }
fi

printf 'os=%s_%s\n' "${os_id}" "${os_version}"
printf 'gpu=%s driver=%s compute_capability=%s total_vram_mib=%s free_vram_mib=%s\n' \
  "${gpu_name}" "${driver_version}" "${compute_capability}" \
  "${memory_total_mib}" "${memory_free_mib}"
printf 'free_vram_policy=install_not_blocked recommended_mib=%s required_mib=%s\n' \
  "${warn_free_vram_mib}" "${require_free_vram_mib}"
if (( memory_free_mib < warn_free_vram_mib )); then
  printf 'warning=low_free_vram actual_mib=%s recommended_mib=%s\n' \
    "${memory_free_mib}" "${warn_free_vram_mib}" >&2
fi
if (( require_free_vram_mib > 0 && memory_free_mib < require_free_vram_mib )); then
  printf 'error=insufficient_free_vram actual_mib=%s required_mib=%s\n' \
    "${memory_free_mib}" "${require_free_vram_mib}" >&2
  exit 1
fi
printf 'rtx4090_preflight=True\n'
