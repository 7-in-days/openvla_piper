#!/usr/bin/env bash
set -euo pipefail
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
exec env OPENVLA_GPU_PROFILE=rtx6000-ada \
  OPENVLA_INSTALL_ENTRYPOINT=scripts/install_rtx6000_ada.sh \
  "${project_root}/scripts/install_rtx4090.sh" "$@"
