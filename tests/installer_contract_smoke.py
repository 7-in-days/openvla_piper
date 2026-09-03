"""Network/GPU-free contract test for supported Ada GPU installers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts/install_rtx4090.sh"
AUTO_INSTALLER = PROJECT_ROOT / "scripts/install_openvla.sh"
RTX6000_ADA_INSTALLER = PROJECT_ROOT / "scripts/install_rtx6000_ada.sh"
SERVER_REQUIREMENTS = PROJECT_ROOT / "requirements-rtx4090.txt"
RTX6000_ADA_REQUIREMENTS = PROJECT_ROOT / "requirements-rtx6000-ada.txt"
COMMON_REQUIREMENTS = PROJECT_ROOT / "requirements-openvla-ada.txt"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def fixture(root: Path, *, os_version: str = "24.04", gpu_row: str | None = None):
    os_release = root / "os-release"
    os_release.write_text(
        f'ID=ubuntu\nVERSION_ID="{os_version}"\n', encoding="utf-8"
    )
    nvidia_smi = root / "nvidia-smi"
    row = gpu_row or "NVIDIA GeForce RTX 4090, 575.64.03, 8.9, 24564, 19000"
    write_executable(nvidia_smi, f"#!/usr/bin/env bash\nprintf '%s\\n' '{row}'\n")
    conda = root / "conda"
    write_executable(
        conda,
        "#!/usr/bin/env bash\nprintf '%s\\n' 'fake conda must not execute during dry-run' >&2\nexit 99\n",
    )
    oft = root / "OpenVLA OFT source"
    (oft / ".git").mkdir(parents=True)
    (oft / "experiments/robot").mkdir(parents=True)
    (oft / "prismatic/vla").mkdir(parents=True)
    (oft / "pyproject.toml").write_text("[project]\nname='openvla-oft'\n", encoding="utf-8")
    (oft / "experiments/robot/openvla_utils.py").write_text("# fixture\n", encoding="utf-8")
    (oft / "prismatic/vla/constants.py").write_text(
        'SUPPORTED_ROBOT_PLATFORMS=("PIPER",)\n', encoding="utf-8"
    )
    piper = root / "Piper source"
    (piper / ".git").mkdir(parents=True)
    (piper / "piper_bridge").mkdir(parents=True)
    (piper / "piper_bridge/config_piper_bridge.py").write_text(
        'frame_topic: str = "/piper/synced/frame"\n'
        'output_topic: str = "/piper/inference/output"\n',
        encoding="utf-8",
    )
    (piper / "piper_bridge/piper_bridge_robot.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    git = root / "git"
    write_executable(
        git,
        """#!/usr/bin/env bash
args="$*"
case "$args" in
  *"OpenVLA OFT source"*"rev-parse HEAD"*) printf '%s\\n' e4287e94541f459edc4feabc4e181f537cd569a8 ;;
  *"Piper source"*"rev-parse HEAD"*) printf '%s\\n' 51e8c43e686ee7d9169cc1badb026c9286055b72 ;;
  *"apply --reverse --check"*) exit 0 ;;
  *"diff --quiet"*) exit 0 ;;
  *) printf 'unexpected fake git call: %s\\n' "$args" >&2; exit 99 ;;
esac
""",
    )
    lerobot = root / "Existing LeRobot env"
    (lerobot / "bin").mkdir(parents=True)
    write_executable(
        lerobot / "bin/python",
        "#!/usr/bin/env bash\nprintf '%s\\n' 'lerobot_environment_valid=True version=0.6.0'\n",
    )
    return os_release, nvidia_smi, conda, git, oft, piper, lerobot


def run_installer(
    root: Path,
    *,
    os_version: str = "24.04",
    gpu_row: str | None = None,
    conda_missing: bool = False,
    installer: Path = INSTALLER,
) -> subprocess.CompletedProcess[str]:
    os_release, nvidia_smi, conda, git, oft, piper, lerobot = fixture(
        root, os_version=os_version, gpu_row=gpu_row
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "OPENVLA_OS_RELEASE_FILE": str(os_release),
            "NVIDIA_SMI_BIN": str(nvidia_smi),
            "CONDA_EXE": str(conda),
            "GIT_BIN": str(git),
            # Deliberately invalid env values prove CLI wins.
            "OPENVLA_INSTALL_PREFIX": "/environment/must/not/win",
            "OPENVLA_INSTALL_PYTHON": "3.10",
            "OPENVLA_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
            "OPENVLA_OFT_REPO": "/environment/oft/must/not/win",
            "PIPER_REPO": "/environment/piper/must/not/win",
        }
    )
    if conda_missing:
        environment.pop("CONDA_EXE")
        environment["PATH"] = "/usr/bin:/bin"
    return subprocess.run(
        [
            "bash",
            str(installer),
            "--dry-run",
            "--skip-ros-check",
            "--prefix",
            str(root / "target prefix"),
            "--python",
            "3.11",
            "--cuda-index",
            "https://download.pytorch.org/whl/cu126",
            "--openvla-oft-repo",
            str(oft),
            "--piper-repo",
            str(piper),
            "--lerobot-prefix",
            str(lerobot),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("installer smoke requires CUDA_VISIBLE_DEVICES='' exactly")
    assert "torch" not in __import__("sys").modules

    requirement_text = COMMON_REQUIREMENTS.read_text(encoding="utf-8")
    rtx4090_requirement_text = SERVER_REQUIREMENTS.read_text(encoding="utf-8")
    rtx6000_requirement_text = RTX6000_ADA_REQUIREMENTS.read_text(encoding="utf-8")
    installer_text = INSTALLER.read_text(encoding="utf-8")
    install_pointer = PROJECT_ROOT / ".install-prefix"
    install_pointer_before = (
        install_pointer.read_bytes() if install_pointer.exists() else None
    )
    for required_distribution in (
        "draccus==0.8.0",
        "matplotlib==3.10.9",
        "protobuf==4.25.9",
        "wandb==0.28.0",
        "imageio==2.37.4",
        "uvicorn==0.52.4",
        "fastapi==0.141.1",
        "roslibpy==1.8.1",
    ):
        assert required_distribution in requirement_text, required_distribution
    assert "lerobot==" not in requirement_text.lower()
    assert "-r requirements-openvla-ada.txt" in rtx4090_requirement_text
    assert "-r requirements-openvla-ada.txt" in rtx6000_requirement_text
    assert "export PYTHONNOUSERSITE=1" in installer_text
    assert "torch.cuda.synchronize" in installer_text

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        success = run_installer(root)
        assert success.returncode == 0, success.stderr
        assert "rtx4090_preflight=True" in success.stdout
        assert "python_version=3.11" in success.stdout
        assert "expected_cuda=12.6" in success.stdout
        assert "installer_complete=planned" in success.stdout
        assert "warning=low_free_vram" in success.stderr
        assert not (root / "target prefix").exists()
        install_pointer_after = (
            install_pointer.read_bytes() if install_pointer.exists() else None
        )
        assert install_pointer_after == install_pointer_before

    with tempfile.TemporaryDirectory() as temporary_directory:
        local_miniforge = run_installer(
            Path(temporary_directory), conda_missing=True
        )
        assert local_miniforge.returncode == 0, local_miniforge.stderr
        assert "Miniforge3-26.3.2-2-Linux-x86_64.sh" in local_miniforge.stdout
        assert ".runtime/miniforge" in local_miniforge.stdout

    with tempfile.TemporaryDirectory() as temporary_directory:
        wrong_os = run_installer(Path(temporary_directory), os_version="20.04")
        assert wrong_os.returncode != 0
        assert "error=unsupported_os" in wrong_os.stderr

    with tempfile.TemporaryDirectory() as temporary_directory:
        wrong_vram = run_installer(
            Path(temporary_directory),
            gpu_row="NVIDIA GeForce RTX 4090, 575.64.03, 8.9, 32768, 30000",
        )
        assert wrong_vram.returncode != 0
        assert "error=vram_mismatch" in wrong_vram.stderr

    with tempfile.TemporaryDirectory() as temporary_directory:
        wrong_capability = run_installer(
            Path(temporary_directory),
            gpu_row="NVIDIA GeForce RTX 4090, 575.64.03, 9.0, 24564, 24000",
        )
        assert wrong_capability.returncode != 0
        assert "error=compute_capability_mismatch" in wrong_capability.stderr

    rtx6000_row = (
        "NVIDIA RTX 6000 Ada Generation, 575.64.03, 8.9, 49140, 47000"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        rtx6000 = run_installer(
            Path(temporary_directory),
            gpu_row=rtx6000_row,
            installer=RTX6000_ADA_INSTALLER,
        )
        assert rtx6000.returncode == 0, rtx6000.stderr
        assert "rtx6000_ada_preflight=True" in rtx6000.stdout
        assert "gpu_profile=rtx6000-ada" in rtx6000.stdout
        assert "requirements-rtx6000-ada.txt" in rtx6000.stdout
        assert "installer_complete=planned" in rtx6000.stdout

    with tempfile.TemporaryDirectory() as temporary_directory:
        automatic = run_installer(
            Path(temporary_directory),
            gpu_row=rtx6000_row,
            installer=AUTO_INSTALLER,
        )
        assert automatic.returncode == 0, automatic.stderr
        assert "gpu_profile=rtx6000-ada" in automatic.stdout

    print("installer_cli_over_env=True")
    print("installer_dry_run_no_conda_execution=True")
    print("project_local_miniforge_fallback=True")
    print("lerobot_environment_reused_read_only=True")
    print("openvla_metadata_dependencies_complete=True")
    print("installer_network_accessed=False")
    print("installer_gpu_accessed=False")
    print("rtx4090_contract=name,capability_8.9,approx_24GiB,kernel")
    print("rtx6000_ada_contract=name,capability_8.9,approx_48GiB,kernel")
    print("gpu_profile_auto_detection=True")
    print("free_vram_install_gate=False")
    print("unsupported_os_capability_vram_fail_closed=True")
    print("installer_contract_smoke=True")


if __name__ == "__main__":
    main()
