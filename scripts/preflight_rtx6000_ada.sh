#!/usr/bin/env bash
set -euo pipefail
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
exec "${project_root}/scripts/preflight_openvla_gpu.sh" \
  --gpu-profile rtx6000-ada "$@"
