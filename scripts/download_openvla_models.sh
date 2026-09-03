#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
export PYTHONNOUSERSITE=1

checkpoint_repo="romalab-cbf/openvla_two_block_pnp"
checkpoint_revision="03a7bb7fc3ccee0448b10b85d808dbae4244a7e7"
base_repo="openvla/openvla-7b"
base_revision="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
checkpoint_dir="${project_root}/artifacts/checkpoints/${checkpoint_repo}"
base_dir="${project_root}/artifacts/models/${base_repo}"

runtime_prefix="${OPENVLA_CONDA_PREFIX:-}"
if [[ -z "${runtime_prefix}" && -r "${project_root}/.install-prefix" ]]; then
  IFS= read -r runtime_prefix <"${project_root}/.install-prefix"
fi
hub_cli="${runtime_prefix}/bin/huggingface-cli"
[[ -x "${hub_cli}" ]] || {
  printf '%s\n' 'error=runtime_environment_missing hint=run_scripts/install_rtx4090.sh' >&2
  exit 1
}

mkdir -p "${checkpoint_dir}" "${base_dir}"
download_snapshot() {
  local repo="$1" revision="$2" destination="$3" attempt
  for attempt in 1 2 3 4 5; do
    if "${hub_cli}" download "${repo}" \
      --revision "${revision}" --local-dir "${destination}"; then
      return 0
    fi
    printf 'warning=download_retry repo=%q attempt=%s/5\n' "${repo}" "${attempt}" >&2
    sleep 5
  done
  printf 'error=download_failed repo=%q revision=%s\n' "${repo}" "${revision}" >&2
  return 1
}

download_snapshot "${checkpoint_repo}" "${checkpoint_revision}" "${checkpoint_dir}"
download_snapshot "${base_repo}" "${base_revision}" "${base_dir}"

required=(
  "${checkpoint_dir}/checkpoint_metadata.json"
  "${checkpoint_dir}/dataset_statistics.json"
  "${checkpoint_dir}/action_head.pt"
  "${checkpoint_dir}/proprio_projector.pt"
  "${checkpoint_dir}/lora_adapter/adapter_config.json"
  "${checkpoint_dir}/lora_adapter/adapter_model.safetensors"
  "${base_dir}/config.json"
  "${base_dir}/model.safetensors.index.json"
  "${base_dir}/model-00001-of-00003.safetensors"
  "${base_dir}/model-00002-of-00003.safetensors"
  "${base_dir}/model-00003-of-00003.safetensors"
)
for path in "${required[@]}"; do
  [[ -s "${path}" ]] || {
    printf 'error=model_file_missing path=%q\n' "${path}" >&2
    exit 1
  }
done
if find "${checkpoint_dir}" "${base_dir}" -type f -name '*.incomplete' -print -quit | grep -q .; then
  printf '%s\n' 'error=incomplete_model_download' >&2
  exit 1
fi
checkpoint_file_count="$(find "${checkpoint_dir}" -type f ! -path '*/.cache/*' | wc -l)"
base_file_count="$(find "${base_dir}" -type f ! -path '*/.cache/*' | wc -l)"
checkpoint_bytes="$(find "${checkpoint_dir}" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{ total += $1 } END { print total + 0 }')"
base_bytes="$(find "${base_dir}" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{ total += $1 } END { print total + 0 }')"
[[ "${checkpoint_file_count}" -eq 156 && "${checkpoint_bytes}" -eq 22307333317 ]] || {
  printf 'error=checkpoint_snapshot_mismatch files=%s bytes=%s\n' \
    "${checkpoint_file_count}" "${checkpoint_bytes}" >&2
  exit 1
}
[[ "${base_file_count}" -eq 18 && "${base_bytes}" -eq 15085153882 ]] || {
  printf 'error=base_snapshot_mismatch files=%s bytes=%s\n' \
    "${base_file_count}" "${base_bytes}" >&2
  exit 1
}

printf 'checkpoint=%s revision=%s files=%s bytes=%s\n' \
  "${checkpoint_dir}" "${checkpoint_revision}" "${checkpoint_file_count}" "${checkpoint_bytes}"
printf 'base_model=%s revision=%s files=%s bytes=%s\n' \
  "${base_dir}" "${base_revision}" "${base_file_count}" "${base_bytes}"
printf 'model_download_complete=True\n'
