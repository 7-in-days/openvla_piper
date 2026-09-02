"""Resolve local directories and explicit Hugging Face snapshot references."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


HF_PREFIX = "hf://"


@dataclass(frozen=True)
class ResolvedSource:
    requested: str
    local_path: Path
    kind: str


def is_hf_source(value: str | Path) -> bool:
    return str(value).strip().startswith(HF_PREFIX)


def _split_hf_reference(value: str) -> tuple[str, str | None]:
    reference = value[len(HF_PREFIX) :].strip()
    if "@" in reference:
        repo_id, revision = reference.rsplit("@", 1)
    else:
        repo_id, revision = reference, None
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ValueError(
            "Hugging Face source must be hf://OWNER/REPO or "
            "hf://OWNER/REPO@REVISION"
        )
    if revision == "":
        raise ValueError("Hugging Face revision after @ must not be empty")
    return repo_id, revision


def resolve_source(
    value: str | Path,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool | None = None,
) -> ResolvedSource:
    """Return a local snapshot path; network use is explicit and runtime-only.

    Local paths are always resolved without network. ``hf://`` references use
    ``huggingface_hub.snapshot_download``. Set ``HF_HUB_OFFLINE=1`` or pass
    ``local_files_only=True`` to require an already-cached snapshot.
    """

    requested = str(value).strip()
    if not requested:
        raise ValueError("checkpoint/model source must not be empty")
    if not is_hf_source(requested):
        path = Path(os.path.expandvars(os.path.expanduser(requested))).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"local source directory does not exist: {path}")
        return ResolvedSource(requested=requested, local_path=path, kind="local")

    repo_id, revision = _split_hf_reference(requested)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "hf:// sources require huggingface_hub; install requirements.txt"
        ) from exc
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=offline if local_files_only is None else local_files_only,
    )
    return ResolvedSource(
        requested=requested,
        local_path=Path(snapshot).resolve(),
        kind="huggingface",
    )
