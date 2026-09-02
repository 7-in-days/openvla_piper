"""Deprecated alias; use :mod:`openvla_pipeline.piper_pipeline`."""

from __future__ import annotations

import sys

from openvla_pipeline.piper_pipeline import (
    DEFAULT_TASK,
    PiperOpenVLAPipeline,
    PiperOpenVLATopicNode,
    PiperPipelineConfig,
    TopicNodeSettings,
    parse_settings as _parse_settings,
)


SYNC_HELP = """usage: python -m openvla_pipeline deploy [options]

Synchronous OpenVLA-to-Piper ROS pipeline.

options:
  --config PATH          server/client/safety runtime JSON
  --checkpoint PATH      checkpoint selected by the shell wrapper
  --action-chunk N       expected checkpoint chunk; mismatch is rejected
  --mode {dry-run,live}  dry-run reads observations without publishing actions
  --task TEXT            language instruction
  --max-actions N        maximum action steps
  --server URL           OpenVLA model server
  --rosbridge-url URL    ROS bridge websocket
  --piper-repo PATH      existing Piper integration repository
  --log-root PATH        inference session log directory
  --session-id ID        explicit inference session identifier
  --health-timeout-s S   model server health timeout
  --request-timeout-s S  model inference request timeout
  --no-chunk-diagnostics disable chunk diagnostic topics
"""


def parse_settings(argv: list[str] | None = None) -> TopicNodeSettings:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(SYNC_HELP.rstrip())
        raise SystemExit(0)
    if any(arg in {"--async-prefetch", "--no-async-prefetch"} for arg in args):
        raise SystemExit(
            "async options moved to: openvla-async-pipeline deploy"
        )
    return _parse_settings([*args, "--no-async-prefetch"])


def main(argv: list[str] | None = None) -> None:
    pipeline = PiperOpenVLAPipeline(parse_settings(argv))
    pipeline.run()

__all__ = [
    "DEFAULT_TASK",
    "PiperOpenVLAPipeline",
    "PiperOpenVLATopicNode",
    "PiperPipelineConfig",
    "TopicNodeSettings",
    "main",
    "parse_settings",
]


if __name__ == "__main__":
    main()
