"""사용자가 직접 수정하는 OpenVLA-Piper 동기 추론 설정.

이 파일의 대문자 값만 바꾸면 된다. 변경값은 다음 process 실행부터 적용된다.
action chunk, action/state dimension, control Hz, camera/action 이름은 checkpoint
metadata에서 자동으로 읽으므로 이 파일에 중복해서 적지 않는다.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── 1. 모델 서버 ──────────────────────────────────────────────────────────────
# 게시된 100K checkpoint와 metadata가 지정한 base revision의 로컬 다운로드 경로.
# scripts/download_openvla_models.sh로 두 저장소 전체를 다운로드한다.
CHECKPOINT: Path | str | None = (
    PROJECT_ROOT / "artifacts/checkpoints/romalab-cbf/openvla_two_block_pnp"
)
BASE_MODEL: Path | str | None = PROJECT_ROOT / "artifacts/models/openvla/openvla-7b"
OPENVLA_OFT_REPO: Path | None = None    # None이면 설치된 openvla-oft package/repository 자동 탐색
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8777
AUTH_TOKEN_ENV = "PIPER_OPENVLA_SERVER_TOKEN"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


# ── 2. 로봇·태스크 ────────────────────────────────────────────────────────────
MODEL_SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ROSBRIDGE_URL = "ws://localhost:9090"
PIPER_REPO = Path.home() / "vla_pipeline"
TASK = "pick up the green blocks one at a time and place them in the white box"
MAX_ACTIONS = 500                       # 최대 실행 tick 수. 20 Hz 기준 500 tick = 25초
HEALTH_TIMEOUT_S = 300.0                   # 7B load + LoRA safe merge startup budget
REQUEST_TIMEOUT_S = 30.0


# ── 3. 로그·진단 ──────────────────────────────────────────────────────────────
INFERENCE_LOG_ROOT = PROJECT_ROOT / "inference_logs"
CHUNK_DIAGNOSTICS = True


# ── 4. 물리 안전 ──────────────────────────────────────────────────────────────
# live 실행은 이 값, PIPER_OPENVLA_LIVE_CONFIRMED=YES, run-robot 명령이 모두 필요하다.
ALLOW_LIVE_MOTION = False
GRIPPER_MIN_M = 0.0
GRIPPER_MAX_M = 0.085
MAX_ARM_STEP_DELTA_RAD = 1.5
