#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
runtime_root="${project_root}/.runtime"

dry_run=0
skip_ros_check=0
disable_conda_bootstrap=0
prefix_cli=""
python_cli=""
cuda_index_cli=""
oft_repo_cli=""
piper_repo_cli=""
lerobot_prefix_cli=""

usage() {
  cat <<'EOF'
usage: scripts/install_rtx4090.sh [options]

Create/update an OpenVLA model-server conda env. The existing LeRobot env is
validated read-only and reused by the client; it is never installed or changed.

options:
  --prefix PATH              OpenVLA server conda prefix
  --python VERSION           3.10 (default) or 3.11
  --cuda-index URL           official PyTorch cu128/cu126 wheel index
  --openvla-oft-repo PATH    already Piper-patched OFT checkout
  --piper-repo PATH          exact PiPER rosbridge adapter checkout
  --lerobot-prefix PATH      existing LeRobot client conda prefix
  --no-conda-bootstrap       fail instead of project-local Miniforge fallback
  --skip-ros-check           skip ROS2 Humble system preflight only
  --dry-run                  validate and print shell-escaped commands
  --help

environment fallbacks:
  OPENVLA_INSTALL_PREFIX, OPENVLA_INSTALL_PYTHON, OPENVLA_TORCH_INDEX_URL,
  OPENVLA_OFT_REPO, PIPER_REPO, LEROBOT_CONDA_PREFIX, active CONDA_PREFIX,
  CONDA_EXE,
  OPENVLA_BOOTSTRAP_CONDA=0|1

precedence: CLI > environment > project-local default
EOF
}

need_value() {
  (( $# >= 2 )) || { printf 'error=%s_requires_a_value\n' "$1" >&2; exit 2; }
}

while (( $# > 0 )); do
  case "$1" in
    --prefix) need_value "$@"; prefix_cli="$2"; shift 2 ;;
    --prefix=*) prefix_cli="${1#--prefix=}"; shift ;;
    --python) need_value "$@"; python_cli="$2"; shift 2 ;;
    --python=*) python_cli="${1#--python=}"; shift ;;
    --cuda-index) need_value "$@"; cuda_index_cli="$2"; shift 2 ;;
    --cuda-index=*) cuda_index_cli="${1#--cuda-index=}"; shift ;;
    --openvla-oft-repo) need_value "$@"; oft_repo_cli="$2"; shift 2 ;;
    --openvla-oft-repo=*) oft_repo_cli="${1#--openvla-oft-repo=}"; shift ;;
    --piper-repo) need_value "$@"; piper_repo_cli="$2"; shift 2 ;;
    --piper-repo=*) piper_repo_cli="${1#--piper-repo=}"; shift ;;
    --lerobot-prefix) need_value "$@"; lerobot_prefix_cli="$2"; shift 2 ;;
    --lerobot-prefix=*) lerobot_prefix_cli="${1#--lerobot-prefix=}"; shift ;;
    --no-conda-bootstrap) disable_conda_bootstrap=1; shift ;;
    --skip-ros-check) skip_ros_check=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'error=unknown_option value=%q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

print_command() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}
run() {
  if (( dry_run == 1 )); then print_command "$@"; else "$@"; fi
}

install_prefix="${prefix_cli:-${OPENVLA_INSTALL_PREFIX:-${runtime_root}/envs/openvla-server}}"
python_version="${python_cli:-${OPENVLA_INSTALL_PYTHON:-3.10}}"
cuda_index="${cuda_index_cli:-${OPENVLA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}}"
oft_explicit=0
piper_explicit=0
if [[ -n "${oft_repo_cli}" ]]; then oft_repo="${oft_repo_cli}"; oft_explicit=1
elif [[ -n "${OPENVLA_OFT_REPO:-}" ]]; then oft_repo="${OPENVLA_OFT_REPO}"; oft_explicit=1
else oft_repo="${runtime_root}/sources/openvla-oft"
fi
if [[ -n "${piper_repo_cli}" ]]; then piper_repo="${piper_repo_cli}"; piper_explicit=1
elif [[ -n "${PIPER_REPO:-}" ]]; then piper_repo="${PIPER_REPO}"; piper_explicit=1
else piper_repo="${runtime_root}/sources/vla-piper"
fi
lerobot_prefix="${lerobot_prefix_cli:-${LEROBOT_CONDA_PREFIX:-${CONDA_PREFIX:-}}}"

for safe_path in "${install_prefix}" "${oft_repo}" "${piper_repo}"; do
  [[ -n "${safe_path}" && "${safe_path}" != "/" ]] || {
    printf 'error=unsafe_path value=%q\n' "${safe_path}" >&2; exit 2;
  }
done
[[ -n "${lerobot_prefix}" && "${lerobot_prefix}" != "/" ]] || {
  printf '%s\n' 'error=lerobot_prefix_required use=--lerobot-prefix_or_LEROBOT_CONDA_PREFIX_or_activate_env' >&2
  exit 2
}
install_prefix="$(realpath -m "${install_prefix}")"
oft_repo="$(realpath -m "${oft_repo}")"
piper_repo="$(realpath -m "${piper_repo}")"
lerobot_prefix="$(realpath -m "${lerobot_prefix}")"
[[ "${python_version}" == "3.10" || "${python_version}" == "3.11" ]] || {
  printf 'error=unsupported_python value=%q allowed=3.10_or_3.11\n' "${python_version}" >&2; exit 2;
}
case "${cuda_index}" in
  https://download.pytorch.org/whl/cu128) expected_torch_cuda="12.8" ;;
  https://download.pytorch.org/whl/cu126) expected_torch_cuda="12.6" ;;
  *) printf 'error=unsupported_cuda_index value=%q allowed=official_cu128_or_cu126\n' "${cuda_index}" >&2; exit 2 ;;
esac

preflight_args=()
if (( skip_ros_check == 1 )); then preflight_args+=(--skip-ros-check); fi
"${project_root}/scripts/preflight_rtx4090.sh" "${preflight_args[@]}"

patch_file="${project_root}/vendor/openvla-oft-piper.patch"
patch_sha="b4008312226af0f1509bd81ccacd13b0306dc845ba767f5a666d8cb170f43e76"
oft_commit="e4287e94541f459edc4feabc4e181f537cd569a8"
piper_commit="51e8c43e686ee7d9169cc1badb026c9286055b72"
oft_url="https://github.com/moojink/openvla-oft.git"
piper_url="https://github.com/7-in-days/vla_pipeline.git"
actual_patch_sha="$(sha256sum "${patch_file}" | awk '{print $1}')"
[[ "${actual_patch_sha}" == "${patch_sha}" ]] || {
  printf 'error=vendor_patch_checksum actual=%s expected=%s\n' "${actual_patch_sha}" "${patch_sha}" >&2; exit 1;
}

git_bin="${GIT_BIN:-}"
if [[ -z "${git_bin}" ]]; then git_bin="$(command -v git || true)"; fi
[[ -n "${git_bin}" && -x "${git_bin}" ]] || {
  printf '%s\n' 'error=git_missing required_for_exact_source_bootstrap' >&2; exit 1;
}

require_file() {
  [[ -f "$1" ]] || { printf 'error=source_contract_missing label=%s path=%q\n' "$2" "$1" >&2; exit 1; }
}

ensure_oft() {
  if [[ ! -d "${oft_repo}/.git" ]]; then
    if (( oft_explicit == 1 )); then
      printf 'error=openvla_oft_not_git_checkout path=%q\n' "${oft_repo}" >&2; exit 1
    fi
    run mkdir -p "$(dirname "${oft_repo}")"
    run "${git_bin}" clone --no-checkout "${oft_url}" "${oft_repo}"
    run "${git_bin}" -C "${oft_repo}" checkout --detach "${oft_commit}"
    run "${git_bin}" -C "${oft_repo}" apply "${patch_file}"
    (( dry_run == 1 )) && return
  fi
  actual_commit="$(${git_bin} -C "${oft_repo}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${oft_commit}" ]] || {
    printf 'error=openvla_oft_commit actual=%s expected=%s path=%q\n' "${actual_commit}" "${oft_commit}" "${oft_repo}" >&2; exit 1;
  }
  if "${git_bin}" -C "${oft_repo}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    :
  elif (( oft_explicit == 0 )) && "${git_bin}" -C "${oft_repo}" apply --check "${patch_file}"; then
    run "${git_bin}" -C "${oft_repo}" apply "${patch_file}"
  else
    printf 'error=openvla_oft_patch_state expected=base_plus_vendor_patch path=%q\n' "${oft_repo}" >&2; exit 1
  fi
  require_file "${oft_repo}/experiments/robot/openvla_utils.py" openvla_oft_runtime
  require_file "${oft_repo}/prismatic/vla/constants.py" openvla_oft_constants
}

ensure_piper() {
  if [[ ! -d "${piper_repo}/.git" ]]; then
    if (( piper_explicit == 1 )); then
      printf 'error=piper_not_git_checkout path=%q\n' "${piper_repo}" >&2; exit 1
    fi
    run mkdir -p "$(dirname "${piper_repo}")"
    run "${git_bin}" clone --no-checkout "${piper_url}" "${piper_repo}"
    run "${git_bin}" -C "${piper_repo}" checkout --detach "${piper_commit}"
    (( dry_run == 1 )) && return
  fi
  actual_commit="$(${git_bin} -C "${piper_repo}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${piper_commit}" ]] || {
    printf 'error=piper_commit actual=%s expected=%s path=%q\n' "${actual_commit}" "${piper_commit}" "${piper_repo}" >&2; exit 1;
  }
  require_file "${piper_repo}/piper_bridge/config_piper_bridge.py" piper_config
  require_file "${piper_repo}/piper_bridge/piper_bridge_robot.py" piper_robot
  "${git_bin}" -C "${piper_repo}" diff --quiet -- \
    piper_bridge/config_piper_bridge.py piper_bridge/piper_bridge_robot.py || {
      printf 'error=piper_runtime_files_modified path=%q\n' "${piper_repo}" >&2; exit 1;
    }
  grep -q 'frame_topic:.*"/piper/synced/frame"' "${piper_repo}/piper_bridge/config_piper_bridge.py"
  grep -q 'output_topic:.*"/piper/inference/output"' "${piper_repo}/piper_bridge/config_piper_bridge.py"
}

ensure_oft
ensure_piper

[[ -x "${lerobot_prefix}/bin/python" ]] || {
  printf 'error=lerobot_python_missing path=%q\n' "${lerobot_prefix}/bin/python" >&2; exit 1;
}
client_probe='import importlib.metadata as m
version=m.version("lerobot")
assert version.startswith("0.6."), version
import numpy, PIL, roslibpy
print(f"lerobot_environment_valid=True version={version}")'
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "${lerobot_prefix}/bin/python" -c "${client_probe}"

bootstrap_conda="${OPENVLA_BOOTSTRAP_CONDA:-1}"
if (( disable_conda_bootstrap == 1 )); then bootstrap_conda=0; fi
[[ "${bootstrap_conda}" == "0" || "${bootstrap_conda}" == "1" ]] || {
  printf 'error=OPENVLA_BOOTSTRAP_CONDA value=%q expected=0_or_1\n' "${bootstrap_conda}" >&2; exit 2;
}
conda_exe="${CONDA_EXE:-}"
if [[ -n "${conda_exe}" && ! -x "${conda_exe}" ]]; then
  printf 'error=CONDA_EXE_not_executable path=%q\n' "${conda_exe}" >&2; exit 1
fi
if [[ -z "${conda_exe}" ]]; then conda_exe="$(command -v conda || true)"; fi
if [[ -z "${conda_exe}" ]]; then
  (( bootstrap_conda == 1 )) || { printf '%s\n' 'error=conda_missing bootstrap_disabled=True' >&2; exit 1; }
  miniforge_root="${runtime_root}/miniforge"
  conda_exe="${miniforge_root}/bin/conda"
  if [[ ! -x "${conda_exe}" ]]; then
    miniforge_url="https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-x86_64.sh"
    miniforge_sha="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"
    installer_path="${runtime_root}/Miniforge3-26.3.2-2-Linux-x86_64.sh"
    run mkdir -p "${runtime_root}"
    run curl --fail --location --proto '=https' --tlsv1.2 --output "${installer_path}" "${miniforge_url}"
    if (( dry_run == 1 )); then
      print_command sha256sum --check "expected:${miniforge_sha} file:${installer_path}"
    else
      printf '%s  %s\n' "${miniforge_sha}" "${installer_path}" | sha256sum --check --status
    fi
    run bash "${installer_path}" -b -p "${miniforge_root}"
  fi
fi

export CONDARC="${runtime_root}/condarc"
export CONDA_PKGS_DIRS="${runtime_root}/conda-pkgs"
export PIP_CACHE_DIR="${runtime_root}/pip-cache"
# A prefix Python otherwise still sees ~/.local packages.  Keep the model
# server reproducible and prevent unrelated packages (for example pyzed) from
# contaminating dependency resolution and pip check.
export PYTHONNOUSERSITE=1
printf 'install_prefix=%s\npython_version=%s\n' "${install_prefix}" "${python_version}"
printf 'torch_versions=2.7.1,0.22.1,2.7.1 expected_cuda=%s index=%s\n' "${expected_torch_cuda}" "${cuda_index}"
printf 'openvla_oft_repo=%s\npiper_repo=%s\nlerobot_prefix=%s\n' "${oft_repo}" "${piper_repo}" "${lerobot_prefix}"
printf 'mode=%s\n' "$([[ ${dry_run} -eq 1 ]] && printf dry-run || printf install)"

if [[ ! -x "${install_prefix}/bin/python" ]]; then
  run "${conda_exe}" create --yes --prefix "${install_prefix}" "python=${python_version}" pip setuptools wheel
else
  if (( dry_run == 1 )); then print_command "${install_prefix}/bin/python" --version
  else
    actual_python="$(${install_prefix}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${actual_python}" == "${python_version}" ]] || { printf 'error=python_version_mismatch actual=%s expected=%s\n' "${actual_python}" "${python_version}" >&2; exit 1; }
    printf 'conda_environment_reused=True\n'
  fi
fi

python_bin="${install_prefix}/bin/python"
run "${python_bin}" -m pip install --upgrade pip==25.1.1 setuptools==75.8.0 wheel==0.45.1
run "${python_bin}" -m pip install --index-url "${cuda_index}" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
run "${python_bin}" -m pip install --constraint "${project_root}/constraints-rtx4090.txt" --requirement "${project_root}/requirements-rtx4090.txt"
run "${python_bin}" -m pip install --no-deps --editable "${oft_repo}"
run "${python_bin}" -m pip install --no-deps --editable "${project_root}"
run "${python_bin}" -m pip check

gpu_probe='import sys, torch, torchvision, torchaudio
expected=sys.argv[1]
assert torch.__version__.split("+")[0]=="2.7.1"
assert torchvision.__version__.split("+")[0]=="0.22.1"
assert torchaudio.__version__.split("+")[0]=="2.7.1"
assert torch.version.cuda==expected and torch.cuda.is_available()
name=torch.cuda.get_device_name(0); cap=torch.cuda.get_device_capability(0)
total=torch.cuda.get_device_properties(0).total_memory//(1024*1024)
assert "RTX 4090" in name and cap==(8,9) and 23000<=total<=26000
probe=torch.arange(16, device="cuda:0", dtype=torch.float32).sum()
torch.cuda.synchronize(0)
assert probe.item()==120.0
print(f"torch_cuda_verified=True name={name} capability={cap} total_vram_mib={total} cuda={torch.version.cuda} arch_list={torch.cuda.get_arch_list()} kernel=True")'
run env PYTHONDONTWRITEBYTECODE=1 "${python_bin}" -c "${gpu_probe}" "${expected_torch_cuda}"
run env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  "${python_bin}" "${project_root}/tests/cpu_only_smoke.py"

if (( dry_run == 0 )); then
  for record in ".install-prefix:${install_prefix}" ".lerobot-prefix:${lerobot_prefix}" ".openvla-oft-repo:${oft_repo}" ".piper-repo:${piper_repo}"; do
    record_name="${record%%:*}"; record_value="${record#*:}"
    temporary_record="$(mktemp "${project_root}/${record_name}.XXXXXX")"
    printf '%s\n' "${record_value}" >"${temporary_record}"
    mv -f "${temporary_record}" "${project_root}/${record_name}"
  done
fi
printf 'installer_complete=%s\n' "$([[ ${dry_run} -eq 1 ]] && printf planned || printf true)"
