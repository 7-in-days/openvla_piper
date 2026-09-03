#!/usr/bin/env bash
set -euo pipefail

gpu_profile="${OPENVLA_GPU_PROFILE:-auto}"
skip_ros_check=0
warn_free_vram_mib="${OPENVLA_WARN_FREE_VRAM_MIB:-20000}"
require_free_vram_mib="${OPENVLA_REQUIRE_FREE_VRAM_MIB:-0}"
minimum_driver="${OPENVLA_MIN_NVIDIA_DRIVER:-570.26}"

usage() {
  cat <<'EOF'
usage: scripts/preflight_openvla_gpu.sh [options]

options:
  --gpu-profile PROFILE        auto, rtx4090, or rtx6000-ada
  --skip-ros-check             do not require /opt/ros/humble/setup.bash
  --warn-free-vram-mib N       warning threshold only (default: 20000)
  --require-free-vram-mib N    fail below N MiB; default 0 means no install gate
  --help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --gpu-profile)
      (( $# >= 2 )) || { printf '%s\n' 'error=--gpu-profile requires a value' >&2; exit 2; }
      gpu_profile="$2"
      shift 2
      ;;
    --gpu-profile=*) gpu_profile="${1#--gpu-profile=}"; shift ;;
    --skip-ros-check) skip_ros_check=1; shift ;;
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
    --help|-h) usage; exit 0 ;;
    *)
      printf 'error=unknown_option value=%q\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${gpu_profile}" in auto|rtx4090|rtx6000-ada) ;; *)
  printf 'error=unsupported_gpu_profile value=%q allowed=auto,rtx4090,rtx6000-ada\n' \
    "${gpu_profile}" >&2
  exit 2
  ;;
esac
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
if [[ -z "${nvidia_smi}" ]]; then nvidia_smi="$(command -v nvidia-smi || true)"; fi
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
profile_for_name() {
  case "$1" in
    *"RTX 6000 Ada"*) printf '%s' rtx6000-ada ;;
    *"RTX 4090"*) printf '%s' rtx4090 ;;
    *) printf '%s' unsupported ;;
  esac
}

detected_profile=""
gpu_name=""
driver_version=""
compute_capability=""
memory_total_mib=""
memory_free_mib=""
while IFS=',' read -r row_name row_driver row_cap row_total row_free; do
  row_name="$(trim "${row_name}")"
  row_profile="$(profile_for_name "${row_name}")"
  if [[ "${row_profile}" == unsupported ]]; then continue; fi
  if [[ "${gpu_profile}" != auto && "${row_profile}" != "${gpu_profile}" ]]; then continue; fi
  detected_profile="${row_profile}"
  gpu_name="${row_name}"
  driver_version="$(trim "${row_driver}")"
  compute_capability="$(trim "${row_cap}")"
  memory_total_mib="$(trim "${row_total}")"
  memory_free_mib="$(trim "${row_free}")"
  break
done <<<"${gpu_rows}"

[[ -n "${detected_profile}" ]] || {
  printf 'error=target_gpu_missing requested_profile=%q supported=RTX_4090_or_RTX_6000_Ada\n' \
    "${gpu_profile}" >&2
  exit 1
}
[[ "${compute_capability}" == "8.9" ]] || {
  printf 'error=compute_capability_mismatch actual=%q expected=8.9 profile=%s\n' \
    "${compute_capability}" "${detected_profile}" >&2
  exit 1
}
[[ "${memory_total_mib}" =~ ^[0-9]+$ ]] || {
  printf 'error=invalid_total_vram value=%q\n' "${memory_total_mib}" >&2
  exit 1
}
case "${detected_profile}" in
  rtx4090) minimum_total_mib=23000; maximum_total_mib=26000 ;;
  rtx6000-ada) minimum_total_mib=45000; maximum_total_mib=51000 ;;
esac
if (( memory_total_mib < minimum_total_mib || memory_total_mib > maximum_total_mib )); then
  printf 'error=vram_mismatch actual_mib=%s expected_mib=%s..%s profile=%s\n' \
    "${memory_total_mib}" "${minimum_total_mib}" "${maximum_total_mib}" \
    "${detected_profile}" >&2
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
printf 'gpu_profile=%s gpu=%s driver=%s compute_capability=%s total_vram_mib=%s free_vram_mib=%s\n' \
  "${detected_profile}" "${gpu_name}" "${driver_version}" "${compute_capability}" \
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
printf 'openvla_gpu_preflight=True profile=%s\n' "${detected_profile}"
if [[ "${detected_profile}" == rtx4090 ]]; then
  printf 'rtx4090_preflight=True\n'
else
  printf 'rtx6000_ada_preflight=True\n'
fi
