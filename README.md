# OpenVLA–PiPER inference workspace

PiPER 로봇에서 OpenVLA-OFT 체크포인트를 추론하기 위한 독립 워크스페이스다. 기본 배포 모델은
[`romalab-cbf/openvla_two_block_pnp`](https://huggingface.co/romalab-cbf/openvla_two_block_pnp)의
100K checkpoint이며, 모델 파일은 git-ignored `artifacts/`에 저장하고 `piper_ws`는 수정하지 않는다.

구성은 세 부분으로 분리된다.

- 이 저장소: LoRA-merged OpenVLA FastAPI 서버, 동기 추론 클라이언트, chunk 진단 및 JSONL 로그
- `/home/pc/vla_pipeline`: `SyncedFrame`을 LeRobot observation으로 변환하는 PiPER 어댑터
- `/home/pc/piper_ws`: 카메라/data hub, local safety gate, executor, rollout rosbag recorder

## 1. 설치

RTX 4090용 OpenVLA 서버 환경은 `.runtime/envs/openvla-server`에 만들고, 기존 LeRobot
환경은 클라이언트용으로 읽기만 한다. 사용자 전역 `~/.local` 패키지는 설치·실행 모두에서
차단된다.

```bash
cd /home/pc/openvla_piper
scripts/install_rtx4090.sh \
  --lerobot-prefix /home/pc/miniconda3/envs/lerobot-060 \
  --piper-repo /home/pc/vla_pipeline
```

설치기는 Ubuntu, RTX 4090, NVIDIA driver, ROS 2 Humble을 검사하고 다음 버전을 고정한다.

- Python 3.10
- PyTorch 2.7.1 / torchvision 0.22.1 / torchaudio 2.7.1, CUDA 12.8 wheel
- OpenVLA-OFT 및 Piper 패치 고정 revision
- `requirements-rtx4090.txt`의 모델 서버 의존성

설치 계획만 확인하려면 `--dry-run`을 추가한다. 정상 설치 후 선택한 경로는
`.install-prefix`, `.lerobot-prefix`, `.openvla-oft-repo`, `.piper-repo`에 기록되며 런처가
자동으로 읽는다.

모델은 별도로 한 번 다운로드한다.

```bash
scripts/openvla-pipeline download-models
```

이 명령은 다음 고정 revision의 저장소 전체를 받는다. checkpoint 저장소에는 root의 최종 100K
파일뿐 아니라 `checkpoints/step-000010000`부터 `step-000100000`까지의 중간 checkpoint와
trainer state도 모두 포함한다.

| 용도 | Hugging Face revision | 로컬 경로 |
|---|---|---|
| fine-tuned checkpoint | `romalab-cbf/openvla_two_block_pnp@03a7bb7fc3ccee0448b10b85d808dbae4244a7e7` | `artifacts/checkpoints/romalab-cbf/openvla_two_block_pnp` |
| base model | `openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f` | `artifacts/models/openvla/openvla-7b` |

## 2. LeRobot 기록을 RLDS로 변환하고 LoRA 학습

LeRobot의 공식 recorder는 현재 `LeRobotDataset v3`(Parquet + MP4)를 기록하며 RLDS를 직접
쓰는 recorder는 제공하지 않는다. OpenVLA와 OpenVLA-OFT의 공식 권장 경로는 수집 데이터를
TFDS/RLDS로 변환한 다음 RLDS loader로 fine-tuning하는 방식이다. 따라서 기존
`vla_pipeline` recorder는 그대로 유지하고 이 저장소에서 변환한다.

원본은 공개된
[`romalab-cbf/two_block_pnp`](https://huggingface.co/datasets/romalab-cbf/two_block_pnp)다.
고정 commit은 `4e1a8ce8da637dca8b2f1437ec5e613186d3dd34`이고 1000 episodes,
699,921 frames, 20 Hz, 약 19.85 GB다. 로컬
`/home/pc/vla_pipeline/episodes/two_block_pnp`의 metadata hash가 이 commit과 일치하므로 다시
다운로드하지 않는다. 변환 manifest에는 repo, commit, 원본 `meta/info.json` SHA-256, split episode
목록을 기록한다.

변환 전용 환경은 LeRobot 환경을 읽기 전용으로 상속하고 TensorFlow/TFDS만 프로젝트 내부에
설치한다.

```bash
cd /home/pc/openvla_piper
scripts/openvla-pipeline install-rlds
```

전체 1000 episodes를 `artifacts/rlds/piper_bridge/1.0.0`으로 변환한다. 5%인 50 episodes는
`val`, 나머지 950개는 `train`으로 episode 단위 분리한다. 원본 35초/700-frame episode를
25초로 자르거나 frame 단위로 섞지 않는다. 출력 경로가 이미 있으면 덮어쓰지 않고 중단한다.

```bash
cd /home/pc/openvla_piper
scripts/openvla-pipeline convert-rlds
scripts/openvla-pipeline verify-rlds
```

변환 전에 1개 episode만 별도 경로에서 시험하려면 다음처럼 실행한다.

```bash
cd /home/pc/openvla_piper
.runtime/envs/rlds-tools/bin/python scripts/convert_lerobot_to_rlds.py \
  --lerobot-root /home/pc/vla_pipeline/episodes/two_block_pnp \
  --output-root /tmp/openvla_piper_rlds_smoke \
  --max-episodes 1 \
  --val-fraction 0

scripts/openvla-pipeline verify-rlds \
  --data-root /tmp/openvla_piper_rlds_smoke \
  --expected-episode-frames 700
```

LoRA fine-tuning은 OpenVLA-OFT의 공식 `torchrun → finetune.py → RLDSDataset` 경로를 사용한다.
PiPER 등록은 third-person + wrist image, 7-D proprio, 7-D absolute joint action, language
instruction이며 FiLM은 단일 instruction 작업이므로 끈다. 기본 action horizon은 요청한 50으로,
20 Hz에서 2.5초 chunk다. 기존 horizon-20 checkpoint의 출력 길이만 50으로 바꾸는 것이 아니라
`PIPER_ACTION_CHUNK=50`으로 action head를 새로 학습한다.

정확한 실행 명령만 먼저 확인:

```bash
scripts/openvla-pipeline train-lora --action-horizon 50 --dry-run
```

RTX 4090에서 실제 학습:

```bash
cd /home/pc/openvla_piper
scripts/openvla-pipeline train-lora \
  --action-horizon 50 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-steps 100000 \
  --save-freq 10000
```

기본 W&B 모드는 `offline`이며, 온라인 기록이 필요할 때만 `--wandb-mode online`을 사용한다.
학습 checkpoint에는 action/state/camera/control-Hz와 원본 LeRobot schema hash를 함께 저장하므로
다른 robot/action contract로 잘못 재개하는 것을 막는다.

공식 레퍼런스는
[LeRobotDataset v3](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx),
[OpenVLA custom dataset 안내](https://github.com/openvla/openvla/blob/main/README.md),
[OpenVLA-OFT ALOHA RLDS/LoRA 절차](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md),
[RLDS builder template](https://github.com/kpertsch/rlds_dataset_builder)다.

## 3. 현재 piper_ws 계약 검증

```bash
cd /home/pc/openvla_piper
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_piper_ws_contract.py
scripts/openvla-pipeline smoke
```

검증 대상은 다음과 같다.

| 역할 | 토픽과 타입 | 계약 |
|---|---|---|
| observation | `/piper/synced/frame` (`piper_msgs/SyncedFrame`) | slave state + wrist/third-person JPEG |
| action | `/piper/inference/output` (`sensor_msgs/JointState`) | 7-D, joint1~6 rad + gripper m, 20 Hz |
| original chunk | `/piper/inference/chunk` (`trajectory_msgs/JointTrajectory`) | 모델의 원본 H×7 |
| aggregated chunk | `/piper/inference/aggregated_chunk` (`trajectory_msgs/JointTrajectory`) | 실행 대기열 N×7 |

chunk publisher와 `piper_ws` recorder는 `RELIABLE / VOLATILE` 계약이다. 물리 동작 enable
토픽 `/piper/inference/ready`는 추론 PC가 아니라 `piper_ws`의 local `inference_gate`가
발행한다.

정적/합성 검증은 ROS 노드나 로봇을 켜지 않는다. 실제 토픽 E2E는 `piper_ws` bring-up 후
다음처럼 확인한다.

```bash
source /opt/ros/humble/setup.bash
source /home/pc/piper_ws/install/setup.bash
ROS_DOMAIN_ID=17 ros2 topic list -t
ROS_DOMAIN_ID=17 ros2 topic hz /piper/synced/frame
```

## 4. 체크포인트, LoRA 병합, FastAPI 계약

`openvla_pipeline/user_settings.py`와 기본 JSON 설정에는 위 로컬 checkpoint, base model,
two-block PnP instruction이 이미 지정돼 있다. 서버 시작 순서는 다음과 같이 고정된다.

1. local-only로 `openvla/openvla-7b` base weights와 processor 로드
2. `PeftModel.from_pretrained()`로 `lora_adapter` 부착
3. `merge_and_unload(safe_merge=True)`로 LoRA를 base weights에 안전 병합
4. `action_head.pt`, `proprio_projector.pt`, dataset statistics 로드
5. 계약 검증이 끝난 단일 policy만 FastAPI/Uvicorn worker 1개로 `/act`에 공개

비동기 action prefetch는 공식 배포 경로에 포함하지 않는다. `/act` 요청과 CUDA 추론은 한 번에
하나씩 동기 처리된다. 서버가 준비된 뒤 `/health`에서 `server_framework=fastapi`,
`inference_mode=synchronous`, `lora_merged=true`를 확인할 수 있다.

```bash
scripts/openvla-pipeline show-config
scripts/openvla-pipeline plan
```

다른 checkpoint를 시험할 때만 `--checkpoint`와 `--base-model`로 덮어쓴다. 설정 우선순위는
CLI > 환경 변수 > JSON config > `user_settings.py`다.

## 5. 추론 명령어

### 모델 서버와 dry-run을 분리 실행

터미널 1:

```bash
cd /home/pc/openvla_piper
scripts/openvla-pipeline model-server \
  --log-level info
```

모델 로드와 LoRA 병합이 끝난 뒤 확인한다.
RTX 4090에서 7B load와 safe merge가 끝날 때까지 수 분 걸릴 수 있으며 자동 실행의 기본
health 대기 한도는 300초다.

```bash
curl --fail http://127.0.0.1:8777/health | python3 -m json.tool
# OpenAPI UI: http://127.0.0.1:8777/docs
```

ROS와 로봇 출력을 쓰지 않고 실제 모델의 1회 추론만 확인하려면 두 번째 터미널에서 실행한다.
checkpoint 통계의 평균 proprio와 256×256 회색 이미지 두 장을 보내 LoRA-merged model,
action head, proprio projector, FastAPI 응답까지 검증한다.

```bash
scripts/openvla-pipeline probe
```

터미널 2:

```bash
cd /home/pc/openvla_piper
scripts/openvla-pipeline dry-run \
  --server http://127.0.0.1:8777 \
  --max-actions 100
```

`dry-run`은 관측과 모델 응답을 검증하지만 `/piper/inference/output` action은 발행하지 않는다.
기본값에서는 두 chunk 진단 토픽과 로그를 남긴다. ROS에도 아무것도 발행하지 않으려면
`--no-chunk-diagnostics`를 추가한다.

로컬 서버를 자동 시작하는 한 줄 실행도 가능하다.

```bash
scripts/openvla-pipeline dry-run \
  --max-actions 100
```

### 실기 추론

먼저 `/home/pc/piper_ws/docs/policy_rollout_bringup.md` 순서대로 camera, rollout recorder,
executor, local `inference_gate`, rosbridge를 준비한다. 실제 action 발행에는 아래 세 조건이
모두 필요하다.

1. `user_settings.py`의 `ALLOW_LIVE_MOTION = True` 또는 JSON config의 동등한 설정
2. 현재 터미널에만 `PIPER_OPENVLA_LIVE_CONFIRMED=YES`
3. 명시적인 `run-robot` 명령

```bash
cd /home/pc/openvla_piper
PIPER_OPENVLA_LIVE_CONFIRMED=YES \
scripts/openvla-pipeline run-robot \
  --max-actions 500
```

confirmation과 bearer token은 `.bashrc`에 저장하지 않는다. 다른 PC의 rosbridge를 쓸 경우
`--rosbridge-url ws://HOST:9090`을 추가하며, 해당 rosbridge와 `piper_ws`는 같은
`ROS_DOMAIN_ID=17`에서 실행해야 한다.

## 6. 로그

각 실행은 `inference_logs/session_<UTC>_<id>/`에 다음을 만든다.

```text
inference_config.json
inference_log.jsonl
inference_observability.jsonl
```

`inference_config.json`은 schema v3 실행 계약, `inference_log.jsonl`은 chunk publish와 연결된
7개 core field, `inference_observability.jsonl`은 lifecycle·timing·failure event다. 단일 bounded
writer thread가 쓰고 종료 시 flush한다. `--no-chunk-diagnostics`이면 core JSONL은 비어 있지만
config와 observability는 계속 기록된다.

자동 시작된 FastAPI 서버의 stdout/stderr는 `artifacts/piper_openvla_topic_server.log`에 남는다.
`model_load_start → base_model_loaded → lora_adapter_attached → lora_adapter_merged →
policy_components_loaded → server_ready` 순서로 시작 상태를 확인할 수 있다.

`piper_ws`의 rollout rosbag은 별도 기록물이다. recorder를 추론보다 먼저 실행해야 late-discovery
이전 chunk 손실을 피할 수 있으며, 기본 위치는
`/home/pc/piper_ws/bag/rollout/sessions/`이다.

현재 세션 로그 확인:

```bash
latest=$(find inference_logs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
python3 -m json.tool "$latest/inference_config.json"
tail -n 5 "$latest/inference_log.jsonl"
tail -n 20 "$latest/inference_observability.jsonl"
```

## 7. 검증 범위

CPU smoke와 contract verifier는 FastAPI `/health`·`/act`·OpenAPI schema, 인증/크기 제한,
설정 우선순위, 안전 gate, action shape/단위, 토픽 이름, chunk 직렬화, JSONL atomic flush를
검증한다. GPU preflight와 설치기는 실제 RTX 4090 CUDA kernel도 실행한다. 실제 checkpoint
load·LoRA 병합·단일 이미지쌍 추론은 모델 서버 검증에 포함하며, `SyncedFrame → OpenVLA → ROS`
dry-run은 camera/data hub와 rosbridge가 실행 중일 때 수행한다.
