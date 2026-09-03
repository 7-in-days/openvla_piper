#!/usr/bin/env bash
set -euo pipefail

# Repository-local bridge to an already running VS Code Git credential provider.
# No token is stored by this script or in .git/config.
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
active_sockets="$(ss -xlHpn 2>/dev/null || true)"
ipc_handle=""
socket_owner_pid=""

while IFS= read -r candidate; do
  if [[ "${active_sockets}" == *"${candidate}"* ]]; then
    ipc_handle="${candidate}"
    socket_line="$(printf '%s\n' "${active_sockets}" | awk -v path="${candidate}" 'index($0, path) {print; exit}')"
    socket_owner_pid="$(printf '%s\n' "${socket_line}" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p')"
    break
  fi
done < <(
  find "${runtime_dir}" -maxdepth 1 -type s -name 'vscode-git-*.sock' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}'
)

if [[ -z "${ipc_handle}" ]]; then
  printf '%s\n' 'error=no_active_vscode_git_credential_socket' >&2
  exit 1
fi

server_root=""
if [[ -n "${socket_owner_pid}" && -x "/proc/${socket_owner_pid}/exe" ]]; then
  node_path="$(readlink "/proc/${socket_owner_pid}/exe")"
  candidate_root="$(dirname "${node_path}")"
  if [[ -x "${candidate_root}/node" && -f "${candidate_root}/extensions/git/dist/askpass-main.js" ]]; then
    server_root="${candidate_root}"
  fi
fi
while IFS= read -r candidate; do
  [[ -z "${server_root}" ]] || break
  if [[ -x "${candidate}/server/node" && -f "${candidate}/server/extensions/git/dist/askpass-main.js" ]]; then
    server_root="${candidate}/server"
    break
  fi
done < <(
  find /home/pc/.vscode-server/cli/servers -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}'
)

if [[ -z "${server_root}" ]]; then
  printf '%s\n' 'error=vscode_git_askpass_not_found' >&2
  exit 1
fi

export VSCODE_GIT_ASKPASS_NODE="${server_root}/node"
export VSCODE_GIT_ASKPASS_MAIN="${server_root}/extensions/git/dist/askpass-main.js"
export VSCODE_GIT_ASKPASS_EXTRA_ARGS=""
export VSCODE_GIT_IPC_HANDLE="${ipc_handle}"
exec "${server_root}/extensions/git/dist/askpass.sh" "$@"
