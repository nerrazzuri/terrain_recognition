"""Runtime topic discovery / verification.

AGENTS.md §3: never trust a roadmap topic name — verify it on the real robot. These helpers
let a node confirm that expected topics exist (and optionally match an expected type)
before subscribing, and **fail closed** when a required topic is absent.

``rclpy`` is imported lazily so importing this module does not require a ROS2 install.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class MissingTopicError(RuntimeError):
    """Raised when a required topic is not present at discovery time."""


@dataclass(frozen=True)
class TopicCheck:
    name: str
    present: bool
    types: tuple[str, ...]
    type_ok: bool


def list_topics(node: object) -> dict[str, list[str]]:
    """Return ``{topic_name: [type, ...]}`` as seen by ``node`` (a live rclpy Node)."""
    return {name: list(types) for name, types in node.get_topic_names_and_types()}


def check_topic(node: object, name: str, expected_type: str | None = None) -> TopicCheck:
    """Check presence (and optionally type) of a single topic against live discovery."""
    available = list_topics(node)
    types = tuple(available.get(name, ()))
    present = name in available
    type_ok = expected_type is None or expected_type in types
    return TopicCheck(name=name, present=present, types=types, type_ok=type_ok)


def require_topics(
    node: object,
    names: Iterable[str],
    expected_types: dict[str, str] | None = None,
) -> list[TopicCheck]:
    """Verify every required topic is present (and type-correct if specified).

    Returns the per-topic checks on success; raises :class:`MissingTopicError` listing the
    failures otherwise — so a node refuses to start against an unverified interface.
    """
    expected_types = expected_types or {}
    checks = [check_topic(node, n, expected_types.get(n)) for n in names]
    failures = [
        c.name for c in checks if not c.present or not c.type_ok
    ]
    if failures:
        raise MissingTopicError(
            "required topics missing or wrong type: " + ", ".join(sorted(failures))
        )
    return checks
