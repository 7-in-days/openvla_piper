#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"

data_root="${project_root}/artifacts/rlds"
run_root="${project_root}/artifacts/training"
base_model="${project_root}/artifacts/models/openvla/openvla-7b"
source_schema="/home/pc/vla_pipeline/episodes/two_block_pnp/meta/info.json"
robot_contract="${project_root}/configs/training/piper_robot_contract.json"
action_horizon=50
batch_size=1
gradient_accumulation=8
max_steps=100000
save_freq=10000
val_freq=10000
lora_rank=32
wandb_mode=offline
dry_run=0

usage() {
  cat <<'EOF'
usage: scripts/train_openvla_lora.sh [options]

  --data-root PATH              TFDS root containing piper_bridge/1.0.0
  --run-root PATH               checkpoint/log output root
  --base-model PATH             local openvla-7b snapshot
  --source-schema PATH          source LeRobot meta/info.json
  --action-horizon N            future absolute actions to train (default: 50)
  --batch-size N                per-GPU batch size (default: 1)
  --gradient-accumulation N     micro-batches per optimizer step (default: 8)
  --max-steps N                 optimizer steps (default: 100000)
  --save-freq N                 checkpoint interval (default: 10000)
  --wandb-mode MODE             offline, online, or disabled (default: offline)
  --dry-run                     print the exact command without starting CUDA training
EOF
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error=invalid_positive_integer name=%s value=%q\n' "${name}" "${value}" >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --base-model) base_model="$2"; shift 2 ;;
    --source-schema) source_schema="$2"; shift 2 ;;
    --action-horizon) action_horizon="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --gradient-accumulation) gradient_accumulation="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --save-freq) save_freq="$2"; shift 2 ;;
    --wandb-mode) wandb_mode="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error=unknown_option value=%q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for pair in \
  "action_horizon:${action_horizon}" \
  "batch_size:${batch_size}" \
  "gradient_accumulation:${gradient_accumulation}" \
  "max_steps:${max_steps}" \
  "save_freq:${save_freq}"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done
case "${wandb_mode}" in offline|online|disabled) ;; *)
  printf 'error=invalid_wandb_mode value=%q\n' "${wandb_mode}" >&2; exit 2 ;;
esac

runtime_prefix="${OPENVLA_CONDA_PREFIX:-}"
oft_repo="${OPENVLA_OFT_REPO:-}"
if [[ -z "${runtime_prefix}" && -r "${project_root}/.install-prefix" ]]; then
  IFS= read -r runtime_prefix <"${project_root}/.install-prefix"
fi
if [[ -z "${oft_repo}" && -r "${project_root}/.openvla-oft-repo" ]]; then
  IFS= read -r oft_repo <"${project_root}/.openvla-oft-repo"
fi

[[ -x "${runtime_prefix}/bin/torchrun" ]] || {
  printf '%s\n' 'error=openvla_runtime_missing hint=run_scripts/install_rtx4090.sh' >&2; exit 1;
}
[[ -f "${oft_repo}/vla-scripts/finetune.py" ]] || {
  printf '%s\n' 'error=openvla_oft_source_missing hint=run_scripts/install_rtx4090.sh' >&2; exit 1;
}
[[ -f "${data_root}/piper_bridge/1.0.0/dataset_info.json" ]] || {
  printf 'error=rlds_dataset_missing path=%q hint=run_scripts/openvla-pipeline_convert-rlds\n' "${data_root}" >&2; exit 1;
}
[[ -f "${base_model}/config.json" ]] || {
  printf 'error=base_model_missing path=%q hint=run_scripts/openvla-pipeline_download-models\n' "${base_model}" >&2; exit 1;
}
[[ -f "${source_schema}" ]] || {
  printf 'error=source_schema_missing path=%q\n' "${source_schema}" >&2; exit 1;
}
[[ -f "${robot_contract}" ]] || {
  printf 'error=robot_contract_missing path=%q\n' "${robot_contract}" >&2; exit 1;
}

mkdir -p "${run_root}" "${project_root}/.runtime/huggingface-training"
command=(
  "${runtime_prefix}/bin/torchrun" --standalone --nnodes 1 --nproc-per-node 1
  "${oft_repo}/vla-scripts/finetune.py"
  --vla_path "${base_model}"
  --data_root_dir "${data_root}"
  --dataset_name piper_bridge
  --control_hz 20
  --robot_contract_path "${robot_contract}"
  --source_schema_path "${source_schema}"
  --run_root_dir "${run_root}"
  --shuffle_buffer_size 10000
  --use_l1_regression True
  --use_diffusion False
  --use_film False
  --num_images_in_input 2
  --use_proprio True
  --batch_size "${batch_size}"
  --learning_rate 5e-4
  --lr_warmup_steps 0
  --num_steps_before_decay 50000
  --grad_accumulation_steps "${gradient_accumulation}"
  --max_steps "${max_steps}"
  --use_val_set True
  --val_freq "${val_freq}"
  --save_freq "${save_freq}"
  --save_latest_checkpoint_only False
  --image_aug True
  --lora_rank "${lora_rank}"
  --lora_dropout 0.0
  --merge_lora_during_training True
  --wandb_entity piper-local
  --wandb_project openvla-piper
  --run_id_note "piper--${action_horizon}_actions--20hz--two_images--proprio--lora"
)

printf 'training_contract=ready dataset=piper_bridge action_horizon=%s control_hz=20 chunk_seconds=%s effective_batch=%s\n' \
  "${action_horizon}" "$(awk -v h="${action_horizon}" 'BEGIN { printf "%.2f", h / 20 }')" \
  "$(( batch_size * gradient_accumulation ))"
if (( dry_run )); then
  printf 'env ROBOT_PLATFORM=PIPER PIPER_ACTION_CHUNK=%q WANDB_MODE=%q TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONNOUSERSITE=1 HF_HOME=%q HF_DATASETS_CACHE=%q ' \
    "${action_horizon}" "${wandb_mode}" \
    "${project_root}/.runtime/huggingface-training" \
    "${project_root}/.runtime/huggingface-training/datasets"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

export ROBOT_PLATFORM=PIPER
export PIPER_ACTION_CHUNK="${action_horizon}"
export WANDB_MODE="${wandb_mode}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONNOUSERSITE=1
export HF_HOME="${project_root}/.runtime/huggingface-training"
export HF_DATASETS_CACHE="${project_root}/.runtime/huggingface-training/datasets"
exec "${command[@]}"
