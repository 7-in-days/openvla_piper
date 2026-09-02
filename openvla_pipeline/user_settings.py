"""사용자가 직접 수정하는 OpenVLA-Piper 동기 추론 설정.

이 파일의 대문자 값만 바꾸면 된다. 변경값은 다음 process 실행부터 적용된다.
action chunk, action/state dimension, control Hz, camera/action 이름은 checkpoint
metadata에서 자동으로 읽으므로 이 파일에 중복해서 적지 않는다.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── 1. 모델 서버 ──────────────────────────────────────────────────────────────
# Dataset/run이 바뀌어도 이 파일이 과거 checkpoint를 조용히 선택하지 않게 한다.
# 실행 시 wrapper의 --checkpoint SOURCE나 PIPER_OPENVLA_CHECKPOINT로 직접 지정한다.
# SOURCE는 로컬 디렉터리 또는 hf://OWNER/REPO@REVISION 형식이다.
CHECKPOINT: str | None = None
BASE_MODEL: str | None = None           # None이면 checkpoint metadata의 base_vla_path 사용
OPENVLA_OFT_REPO: Path | None = None    # None이면 설치된 openvla-oft package/repository 자동 탐색
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8777
AUTH_TOKEN_ENV = "PIPER_OPENVLA_SERVER_TOKEN"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


# ── 2. 로봇·태스크 ────────────────────────────────────────────────────────────
MODEL_SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ROSBRIDGE_URL = "ws://localhost:9090"
PIPER_REPO = Path.home() / "vla-piper"
TASK = "pick up the green blocks one at a time and place them in the white box"
MAX_ACTIONS = 500                       # 최대 실행 tick 수. 20 Hz 기준 500 tick = 25초
HEALTH_TIMEOUT_S = 30.0
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
