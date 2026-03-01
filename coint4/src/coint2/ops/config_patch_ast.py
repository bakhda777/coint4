"""ConfigPatch AST utilities (pair-crypto QuantaAlpha parity adaptation).

This module provides a *structural* intermediate representation over config edits:
- inputs: a list of factor-like patch operations (target_key + op)
- output: an immutable AST

It also implements:
- complexity metrics (symbolic length / parameter count / feature count)
- redundancy metrics via largest common isomorphic subtree (AST matching)

The implementation is intentionally deterministic and small-tree oriented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PatchAstNode:
    label: str
    children: tuple["PatchAstNode", ...] = ()

    def __post_init__(self) -> None:
        token = str(self.label or "").strip()
        if not token:
            raise ValueError("PatchAstNode.label must be non-empty")
        object.__setattr__(self, "label", token)
        object.__setattr__(self, "children", tuple(self.children or ()))


def ast_from_factors(factors: Sequence[Mapping[str, Any]]) -> PatchAstNode:
    """Build a ConfigPatch AST from factor-like operations.

    Expected fields per factor:
    - target_key (dotted key)
    - op (string)

    Tree shape (canonical):
      PATCH
        <root>
          <subkey>...
            op:<op>

    Children are stored in lexicographic label order to make the representation stable.
    """

    root = _MutableNode("PATCH")
    for factor in factors:
        target_key = str(factor.get("target_key") or "").strip()
        op = str(factor.get("op") or "").strip()
        if not target_key or "." not in target_key:
            raise ValueError(f"invalid target_key: {target_key!r}")
        if not op:
            raise ValueError("factor.op must be non-empty")
        parts = [part for part in target_key.split(".") if part]
        if len(parts) < 2:
            raise ValueError(f"invalid target_key: {target_key!r}")
        node = root
        for part in parts:
            node = node.child(part)
        node.child(f"op:{op}")
    return root.freeze()


def symbolic_length(ast: PatchAstNode) -> int:
    """Symbolic length SL(f): number of AST nodes."""

    count = 1
    for child in ast.children:
        count += symbolic_length(child)
    return count


def feature_count(factors: Sequence[Mapping[str, Any]]) -> int:
    """|F_f|: number of distinct target keys touched."""

    out: set[str] = set()
    for factor in factors:
        key = str(factor.get("target_key") or "").strip()
        if key:
            out.add(key)
    return len(out)


def parameter_count(factors: Sequence[Mapping[str, Any]]) -> int:
    """PC(f): number of numeric free parameters in factors.

    For pair-crypto config patches we treat numeric `value` as free parameter for:
    - set/scale/offset
    enable/disable do not carry numeric free parameters.
    """

    total = 0
    for factor in factors:
        op = str(factor.get("op") or "").strip()
        value = factor.get("value")
        if op not in {"set", "scale", "offset"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            total += 1
    return total


def complexity_score(
    *,
    ast: PatchAstNode,
    factors: Sequence[Mapping[str, Any]],
    alpha_sl: float = 1.0,
    alpha_pc: float = 1.0,
    alpha_feat: float = 1.0,
) -> float:
    """Compute complexity score C(f) adapted from QuantaAlpha paper.

    C(f) = alpha_sl * SL(f) + alpha_pc * PC(f) + alpha_feat * log(1 + |F_f|)
    """

    sl = float(symbolic_length(ast))
    pc = float(parameter_count(factors))
    feat = float(feature_count(factors))
    return float(alpha_sl) * sl + float(alpha_pc) * pc + float(alpha_feat) * math.log(1.0 + feat)


def redundancy_similarity(a: PatchAstNode, b: PatchAstNode) -> float:
    """Structural similarity in [0,1] based on max common subtree size.

    similarity = max_common_subtree_size(a, b) / min(size(a), size(b))
    """

    size_a = symbolic_length(a)
    size_b = symbolic_length(b)
    denom = float(min(size_a, size_b))
    if denom <= 0.0:
        return 0.0
    common = float(max_common_subtree_size(a, b))
    return float(min(1.0, max(0.0, common / denom)))


def max_common_subtree_size(a: PatchAstNode, b: PatchAstNode) -> int:
    """Size of the largest common isomorphic subtree between trees a and b.

    This matches the QuantaAlpha notion of AST redundancy:
    the common subtree may be rooted at *any* nodes within the two trees.
    """

    nodes_a = _collect_nodes(a)
    nodes_b = _collect_nodes(b)
    best = 0
    for na in nodes_a:
        for nb in nodes_b:
            best = max(best, _rooted_common_size(na, nb))
    return int(best)


def _collect_nodes(ast: PatchAstNode) -> list[PatchAstNode]:
    out: list[PatchAstNode] = []

    def _walk(node: PatchAstNode) -> None:
        out.append(node)
        for child in node.children:
            _walk(child)

    _walk(ast)
    return out


@lru_cache(maxsize=20000)
def _rooted_common_size(a: PatchAstNode, b: PatchAstNode) -> int:
    if a.label != b.label:
        return 0
    if not a.children or not b.children:
        return 1

    scores: list[list[int]] = []
    for ca in a.children:
        row: list[int] = []
        for cb in b.children:
            row.append(_rooted_common_size(ca, cb))
        scores.append(row)

    return 1 + _max_weight_matching(scores)


def _max_weight_matching(scores: Sequence[Sequence[int]]) -> int:
    """Max sum matching where each column used at most once.

    Intended for very small matrices (tree node degrees).
    """

    if not scores:
        return 0
    left = len(scores)
    right = len(scores[0]) if scores[0] else 0
    if right <= 0:
        return 0
    for row in scores:
        if len(row) != right:
            raise ValueError("invalid score matrix")

    @lru_cache(maxsize=None)
    def rec(i: int, used_mask: int) -> int:
        if i >= left:
            return 0
        best = rec(i + 1, used_mask)
        for j in range(right):
            if used_mask & (1 << j):
                continue
            score = int(scores[i][j])
            if score <= 0:
                continue
            best = max(best, score + rec(i + 1, used_mask | (1 << j)))
        return best

    if right > 20:
        # Defensive fallback: degrees above this are unexpected in our small AST.
        raise ValueError("matching matrix too wide")
    return int(rec(0, 0))


class _MutableNode:
    def __init__(self, label: str) -> None:
        self.label = str(label)
        self.children: dict[str, "_MutableNode"] = {}

    def child(self, label: str) -> "_MutableNode":
        token = str(label)
        if token not in self.children:
            self.children[token] = _MutableNode(token)
        return self.children[token]

    def freeze(self) -> PatchAstNode:
        ordered = tuple(child.freeze() for _, child in sorted(self.children.items(), key=lambda kv: kv[0]))
        return PatchAstNode(label=self.label, children=ordered)

