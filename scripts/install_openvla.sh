#!/usr/bin/env bash
set -euo pipefail
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
exec env OPENVLA_GPU_PROFILE=auto OPENVLA_INSTALL_ENTRYPOINT=scripts/install_openvla.sh \
  "${project_root}/scripts/install_rtx4090.sh" "$@"
