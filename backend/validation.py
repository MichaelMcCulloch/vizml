"""Parameter validation shared by spatial and sequence operators."""

from math import isfinite

from .schema import GraphNode


def integer(node: GraphNode, name: str, default: int, low: int = 1, high: int = 65535) -> int:
    value = node.data.params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be an integer from {low:,} to {high:,}.")
    return value


def real(node: GraphNode, name: str, default: float, low: float, high: float) -> float:
    value = node.data.params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number from {low} to {high}.")
    return float(value)


def choice(node: GraphNode, name: str, default: str, options: set[str]) -> str:
    value = node.data.params.get(name, default)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be one of: {', '.join(sorted(options))}.")
    return value
