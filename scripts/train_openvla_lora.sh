#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${script_path}")/.." && pwd)"

# ============================== USER SETTINGS ==============================
# vla_pipeline scripts처럼 일반적으로 바꾸는 값은 이 블록에 모아 둔다.
# 아래 CLI 옵션을 주면 해당 실행에 한해 이 값보다 우선한다.
DATASET_NAME="piper_bridge"                         # OpenVLA-OFT registry name
DATA_ROOT="${project_root}/artifacts/rlds"         # contains DATASET_NAME/1.0.0
RUN_ROOT="${project_root}/artifacts/training"
BASE_MODEL="${project_root}/artifacts/models/openvla/openvla-7b"
SOURCE_SCHEMA="/home/pc/vla_pipeline/episodes/two_block_pnp/meta/info.json"
ROBOT_CONTRACT="${project_root}/configs/training/piper_robot_contract.json"

CONTROL_HZ=20
ACTION_HORIZON=50                                   # 50 / 20 Hz = 2.5 seconds
NUM_IMAGES_IN_INPUT=2                               # third_person + wrist
USE_PROPRIO=True
USE_FILM=False                                      # single-instruction task
USE_L1_REGRESSION=True
USE_DIFFUSION=False

BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
SHUFFLE_BUFFER_SIZE=10000
LEARNING_RATE="5e-4"
LR_WARMUP_STEPS=0
NUM_STEPS_BEFORE_DECAY=50000
MAX_STEPS=100000
USE_VAL_SET=True
VAL_FREQ=10000
VAL_TIME_LIMIT=180
SAVE_FREQ=10000
SAVE_LATEST_CHECKPOINT_ONLY=False
IMAGE_AUG=True

LORA_RANK=32
LORA_ALPHA=""                                       # empty = OpenVLA-OFT default
LORA_DROPOUT=0.0
MERGE_LORA_DURING_TRAINING=True

WANDB_MODE=offline                                  # offline | online | disabled
WANDB_ENTITY="piper-local"
WANDB_PROJECT="openvla-piper"
RUN_ID_NOTE=""                                      # empty = generated from settings
# ===========================================================================

data_root="${DATA_ROOT}"
run_root="${RUN_ROOT}"
base_model="${BASE_MODEL}"
source_schema="${SOURCE_SCHEMA}"
robot_contract="${ROBOT_CONTRACT}"
dataset_name="${DATASET_NAME}"
control_hz="${CONTROL_HZ}"
action_horizon="${ACTION_HORIZON}"
batch_size="${BATCH_SIZE}"
gradient_accumulation="${GRADIENT_ACCUMULATION}"
max_steps="${MAX_STEPS}"
save_freq="${SAVE_FREQ}"
val_freq="${VAL_FREQ}"
lora_rank="${LORA_RANK}"
wandb_mode="${WANDB_MODE}"
dry_run=0

usage() {
  cat <<'EOF'
usage: scripts/train_openvla_lora.sh [options]

  --data-root PATH              TFDS root containing piper_bridge/1.0.0
  --run-root PATH               checkpoint/log output root
  --base-model PATH             local openvla-7b snapshot
  --source-schema PATH          source LeRobot meta/info.json
  --dataset-name NAME           OpenVLA-OFT registered RLDS name
  --control-hz HZ               dataset/control frequency
  --action-horizon N            future absolute actions to train (default: 50)
  --batch-size N                per-GPU batch size (default: 1)
  --gradient-accumulation N     micro-batches per optimizer step (default: 8)
  --max-steps N                 optimizer steps (default: 100000)
  --save-freq N                 checkpoint interval (default: 10000)
  --learning-rate RATE          LoRA/action-head learning rate
  --lora-rank N                 LoRA rank
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
    --dataset-name) dataset_name="$2"; shift 2 ;;
    --control-hz) control_hz="$2"; shift 2 ;;
    --action-horizon) action_horizon="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --gradient-accumulation) gradient_accumulation="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --save-freq) save_freq="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --lora-rank) lora_rank="$2"; shift 2 ;;
    --wandb-mode) wandb_mode="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error=unknown_option value=%q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for pair in \
  "action_horizon:${action_horizon}" \
  "control_hz:${control_hz}" \
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
[[ -f "${data_root}/${dataset_name}/1.0.0/dataset_info.json" ]] || {
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
  --dataset_name "${dataset_name}"
  --control_hz "${control_hz}"
  --robot_contract_path "${robot_contract}"
  --source_schema_path "${source_schema}"
  --run_root_dir "${run_root}"
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --use_l1_regression "${USE_L1_REGRESSION}"
  --use_diffusion "${USE_DIFFUSION}"
  --use_film "${USE_FILM}"
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
  --use_proprio "${USE_PROPRIO}"
  --batch_size "${batch_size}"
  --learning_rate "${LEARNING_RATE}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
  --grad_accumulation_steps "${gradient_accumulation}"
  --max_steps "${max_steps}"
  --use_val_set "${USE_VAL_SET}"
  --val_freq "${val_freq}"
  --val_time_limit "${VAL_TIME_LIMIT}"
  --save_freq "${save_freq}"
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
  --image_aug "${IMAGE_AUG}"
  --lora_rank "${lora_rank}"
  --lora_dropout "${LORA_DROPOUT}"
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --run_id_note "${RUN_ID_NOTE:-piper--${action_horizon}_actions--${control_hz}hz--${NUM_IMAGES_IN_INPUT}_images--proprio--lora}"
)

if [[ -n "${LORA_ALPHA}" ]]; then
  command+=(--lora_alpha "${LORA_ALPHA}")
fi

printf 'training_contract=ready dataset=%s action_horizon=%s control_hz=%s chunk_seconds=%s effective_batch=%s\n' \
  "${dataset_name}" "${action_horizon}" "${control_hz}" \
  "$(awk -v h="${action_horizon}" -v hz="${control_hz}" 'BEGIN { printf "%.2f", h / hz }')" \
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
