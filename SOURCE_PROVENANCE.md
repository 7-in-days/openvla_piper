# Source provenance

이 워크스페이스의 OpenVLA-OFT 기반 구현은 다음 고정 source를 사용한다.

- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8` +
  `vendor/openvla-oft-piper.patch`
- PiPER LeRobot/rosbridge adapter: `/home/pc/vla_pipeline`,
  `51e8c43e686ee7d9169cc1badb026c9286055b72`
- PiPER fine-tuned checkpoint: `romalab-cbf/openvla_two_block_pnp`,
  `03a7bb7fc3ccee0448b10b85d808dbae4244a7e7`
- OpenVLA base model: `openvla/openvla-7b`,
  `47a0ec7fc4ec123775a391911046cf33cf9ed83f`

2026-09-03에는 현재 `/home/pc/piper_ws`의 executor 및 rollout recorder 계약도 읽기 전용으로
대조했다. 대조한 workspace HEAD는 `74c35cc`이며, 사용자가 작업 중인 recorder 변경도 현재
worktree 상태 그대로 검증했다.

OpenVLA 서버 환경과 OpenVLA-OFT source는 이 프로젝트의 `.runtime/` 아래에만 설치했다.
`/home/pc/vla_pipeline`과 `/home/pc/piper_ws`는 수정하지 않았다. 모델 파일은 재현 가능한
고정 revision으로 이 저장소의 git-ignored `artifacts/` 아래에만 다운로드한다.
