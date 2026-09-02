# OpenVLA two-block Piper inference

This folder is a standalone **inference-only** copy. It does not contain a
checkpoint and does not modify `/home/roma/vla-piper` or the OpenVLA training
workspace.

## 1. Install

Keep the existing LeRobot environment. Run one idempotent installer to create a
separate OpenVLA model-server environment, install the training-compatible
official PyTorch CUDA wheels, validate the RTX 4090, and finish with the
CPU-only smoke:

```bash
cd /path/to/openvla_two_block_pnp
scripts/install_rtx4090.sh --lerobot-prefix /path/to/existing/lerobot-env
```

If the existing LeRobot environment is currently active, this shorter form
uses its `CONDA_PREFIX`:

```bash
conda activate YOUR_EXISTING_LEROBOT_ENV
scripts/install_rtx4090.sh
```

The installer validates LeRobot `0.6.x`, NumPy, Pillow, and roslibpy in that
prefix without installing, upgrading, or deleting anything there. It clones
the exact OpenVLA-OFT and vla-piper revisions into project-local `.runtime`
and applies the vendored Piper patch. The source-path flags are optional and
are only for already-matching local checkouts.

Preview without conda creation, downloads, GPU Python calls, or file writes:

```bash
scripts/install_rtx4090.sh --dry-run \
  --lerobot-prefix /path/to/existing/lerobot-env
```

The host preflight requires Ubuntu 22.04 or 24.04, an NVIDIA driver compatible
with the pinned CUDA wheel, an RTX 4090 with compute capability 8.9 and about
24 GiB VRAM, plus ROS2 Humble unless `--skip-ros-check` is explicit. The
installer does **not** apt-install a CUDA toolkit or ROS, load a model, download
a checkpoint, connect to ROS, or publish actions.

Defaults are Python 3.10, `torch==2.7.1`, `torchvision==0.22.1`,
`torchaudio==2.7.1`, and the official
`https://download.pytorch.org/whl/cu128` index. Override supported installer
values with CLI flags or the environment variables listed by:

```bash
scripts/install_rtx4090.sh --help
```

The successful prefix is stored in project-local `.install-prefix`; launchers
read it automatically. `OPENVLA_CONDA_PREFIX` remains an explicit override.
Before every model-server start, preflight prints total/free VRAM and warns when
free VRAM is below 20000 MiB. Set `OPENVLA_REQUIRE_FREE_VRAM_MIB=20000` to turn
that recommendation into a hard gate.

An explicit `OPENVLA_OFT_REPO` must be the Piper-capable checkout used for
training. The server rejects a wrong checkout, action chunk, action/proprio
dimension, or normalization before loading weights.

Checkpoint sources are not copied into this folder:

```bash
# Local atomic checkpoint directory
--checkpoint /absolute/path/to/step-000100000

# Hugging Face snapshot; @REVISION is strongly recommended
--checkpoint hf://OWNER/REPOSITORY@REVISION
```

`hf://` uses `huggingface_hub.snapshot_download`. Set `HF_HUB_OFFLINE=1` to
require an already-cached snapshot. If the checkpoint metadata contains a base
model path from another machine, pass `--base-model` or `OPENVLA_BASE_MODEL`
with a local directory or another `hf://` source.

For the published 100K model, use the immutable revisions below. The explicit
base model is required because the checkpoint metadata records the training
machine's local cache path:

```bash
--checkpoint hf://romalab-cbf/openvla_two_block_pnp@03a7bb7fc3ccee0448b10b85d808dbae4244a7e7 \
--base-model hf://openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

The repository-root HF source resolves the final 100K checkpoint. To select an
earlier step, download that snapshot once and pass the local
`checkpoints/step-XXXXXXXXX` directory.

## 2. Configure and dry-run

Edit only the uppercase values in
`openvla_pipeline/user_settings.py`, or use the checked JSON example:

```bash
scripts/openvla-pipeline show-config
scripts/openvla-pipeline plan \
  --checkpoint /absolute/path/to/step-000100000 \
  --openvla-oft-repo "$OPENVLA_OFT_REPO"
scripts/openvla-pipeline dry-run \
  --checkpoint /absolute/path/to/step-000100000 \
  --openvla-oft-repo "$OPENVLA_OFT_REPO" \
  --max-actions 100
```

`plan` performs no model load, ROS connection, or GPU access. `dry-run` reads
`/piper/synced/frame` but never publishes `/piper/inference/output`. By default
it still publishes the diagnostic topics below; add `--no-chunk-diagnostics`
for a fully read-only ROS run.

Precedence is **CLI > environment > JSON config > `user_settings.py`**.
The config selector itself follows `--config` >
`PIPER_OPENVLA_RUNTIME_CONFIG` > `user_settings.py`.

## 3. Server and client separately

Terminal 1:

```bash
scripts/openvla-pipeline model-server \
  --checkpoint hf://romalab-cbf/openvla_two_block_pnp@03a7bb7fc3ccee0448b10b85d808dbae4244a7e7 \
  --base-model hf://openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

Terminal 2:

```bash
scripts/openvla-pipeline dry-run \
  --server http://127.0.0.1:8777 \
  --no-chunk-diagnostics \
  --max-actions 100
```

The sync contract is:

- input: `/piper/synced/frame`
- physical output: `/piper/inference/output`
- diagnostics: `/piper/inference/chunk` and
  `/piper/inference/aggregated_chunk`, both
  `trajectory_msgs/JointTrajectory`
- action shape, camera order, robot names/units, control rate, and
  normalization come from checkpoint metadata and are cross-checked against
  the imported OpenVLA-OFT constants and Piper bridge configuration

Each session writes `inference_config.json` schema v3 atomically, an exact
seven-field `inference_log.jsonl` core row only after a diagnostic chunk
publish, and independent lifecycle/timing/failure events in
`inference_observability.jsonl`. One bounded writer thread handles both JSONL
files; native CPU pools are capped by the launcher.

## 4. Live safety gate

Physical output is denied by default. All three conditions are required:

1. Set `ALLOW_LIVE_MOTION = True` in `user_settings.py` or set
   `safety.allow_live_motion` to `true` in an explicit JSON config.
2. Set `PIPER_OPENVLA_LIVE_CONFIRMED=YES` for that terminal only.
3. Run the explicit command `scripts/openvla-pipeline run-robot ...`.

Do not store the confirmation or bearer token in `.bashrc`. The launcher never
prints the token value.

## 5. CPU-only verification

```bash
CUDA_VISIBLE_DEVICES='' scripts/openvla-pipeline smoke
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 tests/installer_contract_smoke.py
```

This uses temporary synthetic metadata and fake ROS/config objects. It does not
load Torch/TensorFlow, access the network, connect to rosbridge, or publish a
robot command.

These checks prove configuration, metadata, logging, topic, installer, and
safety contracts. Full end-to-end status remains unverified until the target
RTX 4090 completes a real checkpoint load, one inference, and a synchronous
ROS dry-run.
