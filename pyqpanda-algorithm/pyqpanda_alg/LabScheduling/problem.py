"""Validated, immutable inputs for shared laboratory rescheduling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Resource:
    """A unit-capacity instrument with cleanup time and half-open downtime."""

    id: str
    cleanup: int
    downtime: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Option:
    """An explicitly allowed instrument/start assignment and preference cost."""

    resource: str
    start: int
    cost: int


@dataclass(frozen=True)
class Task:
    """An indivisible experiment belonging to one mutually exclusive group."""

    id: str
    group: str
    duration: int
    options: tuple[Option, ...]
    original: tuple[str, int] | None
    change_cost: int


@dataclass(frozen=True)
class Problem:
    """Validated finite scheduling instance; empty domains may be infeasible."""

    name: str
    horizon: int
    resources: tuple[Resource, ...]
    tasks: tuple[Task, ...]
    precedence: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        """Parse strict JSON-shaped data; reject typos, booleans and bad references.

        Costs and time values are bounded nonnegative integers. A task with no
        options is a valid but infeasible instance, as is a cyclic precedence
        graph (durations are strictly positive).
        """
        _keys(
            data, {"name", "horizon", "resources", "tasks"}, {"precedence"}, "problem"
        )
        name = _name(data["name"], "name")
        horizon = _integer(data["horizon"], "horizon", minimum=1)
        resources = []
        for raw in _list(data["resources"], "resources"):
            _keys(raw, {"id"}, {"cleanup", "downtime"}, "resource")
            rid = _name(raw["id"], "resource.id")
            cleanup = _integer(raw.get("cleanup", 0), "cleanup")
            downtime = []
            for interval in _list(raw.get("downtime", []), "downtime"):
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError("downtime must contain [start, end] pairs")
                start, end = (_integer(v, "downtime endpoint") for v in interval)
                if not 0 <= start < end <= horizon:
                    raise ValueError(
                        "downtime must satisfy 0 <= start < end <= horizon"
                    )
                downtime.append((start, end))
            resources.append(Resource(rid, cleanup, tuple(downtime)))
        _unique([r.id for r in resources], "resource ids")
        if not resources:
            raise ValueError("at least one resource is required")
        resource_ids = {r.id for r in resources}
        tasks = []
        for raw in _list(data["tasks"], "tasks"):
            _keys(
                raw,
                {"id", "group", "duration", "options"},
                {"original", "change_cost"},
                "task",
            )
            tid = _name(raw["id"], "task.id")
            group = _name(raw["group"], "group")
            duration = _integer(raw["duration"], "duration", minimum=1)
            options = []
            for option in _list(raw["options"], "options"):
                _keys(option, {"resource", "start", "cost"}, set(), "option")
                rid, start = _assignment(option, resource_ids)
                options.append(Option(rid, start, _integer(option["cost"], "cost")))
            _unique([(o.resource, o.start) for o in options], "task options")
            original = None
            if "original" in raw:
                _keys(raw["original"], {"resource", "start"}, set(), "original")
                original = _assignment(raw["original"], resource_ids)
            change_cost = _integer(raw.get("change_cost", 0), "change_cost")
            if change_cost and original is None:
                raise ValueError("change_cost requires original")
            tasks.append(
                Task(tid, group, duration, tuple(options), original, change_cost)
            )
        _unique([t.id for t in tasks], "task ids")
        task_ids = {t.id for t in tasks}
        precedence = []
        for pair in _list(data.get("precedence", []), "precedence"):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("precedence must contain [before, after] pairs")
            before, after = (_name(v, "precedence task") for v in pair)
            if before not in task_ids or after not in task_ids or before == after:
                raise ValueError("precedence needs two distinct known task ids")
            precedence.append((before, after))
        _unique(precedence, "precedence pairs")
        return cls(name, horizon, tuple(resources), tuple(tasks), tuple(precedence))


def load_problem(path: str | Path) -> Problem:
    """Read UTF-8 JSON, rejecting duplicate keys and non-finite constants."""
    return parse_problem(Path(path).read_text(encoding="utf-8"))


def parse_problem(text: str) -> Problem:
    """Parse a JSON snapshot with the same strict validation as file loading."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate object keys instead of silently keeping the last."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        """Reject nonstandard JSON numeric constants."""
        raise ValueError(f"invalid JSON constant: {value}")

    return Problem.from_dict(
        json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=invalid_constant,
        )
    )


def _keys(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"{label}: missing or unknown fields")


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 1_000_000:
        raise ValueError(f"{label} must be an integer in [{minimum}, 1000000]")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _unique(values: list[Any], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


def _assignment(raw: dict[str, Any], resource_ids: set[str]) -> tuple[str, int]:
    rid = _name(raw["resource"], "resource reference")
    if rid not in resource_ids:
        raise ValueError(f"unknown resource: {rid}")
    return rid, _integer(raw["start"], "start")
