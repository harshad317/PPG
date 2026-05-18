"""
PPG core graph objects: PromptFragment, Guard, PPGraph, GraphValidator.

Topology is FROZEN after construction — only guard weights and node utility
scores are learned during training. This keeps the optimization problem bounded
and the regret bound clean (routing layer only).
"""

from __future__ import annotations

import collections
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

import numpy as np

# features.py owns FEATURE_NAMES/FEATURE_DIM — import here to keep Guard aligned.
# Import is deferred inside functions that need it to avoid circular import at
# module load time (features.py has no dependency on graph.py).
from ppg.core.features import FEATURE_NAMES, FEATURE_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Fragment types
# ---------------------------------------------------------------------------

class FragmentType(str, Enum):
    TASK_FRAMING           = "task_framing"
    REASONING_STYLE        = "reasoning_style"
    OUTPUT_CONTRACT        = "output_contract"
    UNCERTAINTY_ESCALATION = "uncertainty_escalation"
    VERIFICATION           = "verification"
    TOOL_USE               = "tool_use"
    COMPRESSION            = "compression"
    DOMAIN_PRIMER          = "domain_primer"


# Structural constraints: every valid path must include exactly one of each.
REQUIRED_TYPES: frozenset[FragmentType] = frozenset({
    FragmentType.TASK_FRAMING,
    FragmentType.OUTPUT_CONTRACT,
})


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

@dataclass
class Guard:
    """
    Linear threshold predicate over runtime features.
    g(phi) = 1  iff  weights @ phi >= bias

    Initialized to all-pass (bias = -inf) so the initial policy includes
    all nodes. Bandit training updates weights and bias.
    """
    weights: np.ndarray          # shape (FEATURE_DIM,)
    bias: float = -1e9           # start all-pass

    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())

    def __post_init__(self):
        assert len(self.weights) == len(self.feature_names), (
            f"weights dim {len(self.weights)} != feature_names len {len(self.feature_names)}"
        )

    @classmethod
    def all_pass(cls) -> "Guard":
        return cls(weights=np.zeros(FEATURE_DIM), bias=-1e9)

    def evaluate(self, phi: np.ndarray) -> bool:
        """Returns True if this guard fires (edge is traversable)."""
        return float(self.weights @ phi) >= self.bias

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Guard":
        return cls(
            weights=np.array(d["weights"]),
            bias=d["bias"],
            feature_names=d["feature_names"],
        )


# ---------------------------------------------------------------------------
# PromptFragment
# ---------------------------------------------------------------------------

@dataclass
class PromptFragment:
    """
    One typed node in the PPGraph.

    template: Jinja2-style string with {input} and {context} slots.
    token_count: cached at creation time; updated when template changes.
    utility: running mean of marginal reward from LOO credit assignment.
    """
    id: str
    type: FragmentType
    template: str
    token_count: int = 0
    utility: float = 0.0          # updated by CreditAssigner
    utility_n: int = 0            # number of utility samples seen
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, ftype: FragmentType, template: str, **metadata) -> "PromptFragment":
        return cls(
            id=str(uuid.uuid4()),
            type=ftype,
            template=template,
            metadata=metadata,
        )

    def render(self, context: dict) -> str:
        """Simple {key} substitution — missing keys become empty strings."""
        return self.template.format_map(collections.defaultdict(str, context))

    def update_utility(self, marginal_reward: float) -> None:
        """Online mean update for LOO credit assignment."""
        self.utility_n += 1
        self.utility += (marginal_reward - self.utility) / self.utility_n

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "template": self.template,
            "token_count": self.token_count,
            "utility": self.utility,
            "utility_n": self.utility_n,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptFragment":
        return cls(
            id=d["id"],
            type=FragmentType(d["type"]),
            template=d["template"],
            token_count=d.get("token_count", 0),
            utility=d.get("utility", 0.0),
            utility_n=d.get("utility_n", 0),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# PPGraph
# ---------------------------------------------------------------------------

@dataclass
class PPGraph:
    """
    Typed directed acyclic graph of PromptFragments.

    Topology is frozen after construction. Only guard parameters and
    fragment utility scores are mutable during training.

    nodes:       node_id -> PromptFragment
    edges:       (src_id, dst_id) -> Guard
    source_ids:  nodes with no incoming edges (always traversed first)
    terminal_ids: nodes with no outgoing edges (path ends here)
    """
    nodes: dict[str, PromptFragment]
    edges: dict[tuple[str, str], Guard]
    source_ids: frozenset[str]
    terminal_ids: frozenset[str]

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> list[str]:
        return [dst for (src, dst) in self.edges if src == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        return [src for (src, dst) in self.edges if dst == node_id]

    def guard(self, src: str, dst: str) -> Guard:
        return self.edges[(src, dst)]

    def active_successors(self, node_id: str, phi: np.ndarray) -> list[str]:
        """Successors whose guard fires on feature vector phi."""
        return [
            dst for dst in self.successors(node_id)
            if self.edges[(node_id, dst)].evaluate(phi)
        ]

    def nodes_by_type(self, ftype: FragmentType) -> list[PromptFragment]:
        return [n for n in self.nodes.values() if n.type == ftype]

    # ------------------------------------------------------------------
    # Path enumeration (DFS, used for ablations and visualization)
    # ------------------------------------------------------------------

    def all_paths(self) -> Iterator[list[str]]:
        """Enumerate all source-to-terminal paths (topology only, no guards)."""
        def dfs(current: str, path: list[str]):
            path = path + [current]
            if current in self.terminal_ids:
                yield path
                return
            nexts = self.successors(current)
            if not nexts:
                yield path  # dead end, treat as terminal
                return
            for nxt in nexts:
                yield from dfs(nxt, path)

        for src in self.source_ids:
            yield from dfs(src, [])

    def path_count(self) -> int:
        return sum(1 for _ in self.all_paths())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: frag.to_dict() for nid, frag in self.nodes.items()},
            "edges": {
                f"{src}|{dst}": guard.to_dict()
                for (src, dst), guard in self.edges.items()
            },
            "source_ids": list(self.source_ids),
            "terminal_ids": list(self.terminal_ids),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PPGraph":
        nodes = {nid: PromptFragment.from_dict(fd) for nid, fd in d["nodes"].items()}
        edges = {}
        for key, gd in d["edges"].items():
            src, dst = key.split("|")
            edges[(src, dst)] = Guard.from_dict(gd)
        return cls(
            nodes=nodes,
            edges=edges,
            source_ids=frozenset(d["source_ids"]),
            terminal_ids=frozenset(d["terminal_ids"]),
        )

    @classmethod
    def from_json(cls, path: str) -> "PPGraph":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# GraphValidator
# ---------------------------------------------------------------------------

class GraphValidator:
    """
    Validates a PPGraph before training begins.

    Checks:
      1. Acyclicity (DFS cycle detection)
      2. Required types present (TASK_FRAMING, OUTPUT_CONTRACT)
      3. Every path from source reaches a terminal
      4. No isolated nodes (every non-source has at least one predecessor)
      5. Edge endpoints exist in node set
    """

    def validate(self, g: PPGraph) -> list[str]:
        errors: list[str] = []
        errors += self._check_edge_endpoints(g)
        cycle_errors = self._check_acyclicity(g)
        errors += cycle_errors
        if cycle_errors:
            # Path-based checks require a DAG — skip to avoid infinite recursion.
            return errors
        errors += self._check_required_types(g)
        errors += self._check_reachability(g)
        errors += self._check_no_isolated(g)
        errors += self._check_path_type_coverage(g)
        return errors

    def validate_or_raise(self, g: PPGraph) -> None:
        errors = self.validate(g)
        if errors:
            raise ValueError("PPGraph validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # ------------------------------------------------------------------

    def _check_edge_endpoints(self, g: PPGraph) -> list[str]:
        errors = []
        for src, dst in g.edges:
            if src not in g.nodes:
                errors.append(f"Edge src '{src}' not in nodes")
            if dst not in g.nodes:
                errors.append(f"Edge dst '{dst}' not in nodes")
        return errors

    def _check_acyclicity(self, g: PPGraph) -> list[str]:
        # DFS with WHITE/GRAY/BLACK coloring
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in g.nodes}

        def dfs(nid: str) -> bool:
            color[nid] = GRAY
            for dst in g.successors(nid):
                if color[dst] == GRAY:
                    return True   # cycle found
                if color[dst] == WHITE and dfs(dst):
                    return True
            color[nid] = BLACK
            return False

        for nid in g.nodes:
            if color[nid] == WHITE:
                if dfs(nid):
                    return ["Graph contains a cycle (must be a DAG)"]
        return []

    def _check_required_types(self, g: PPGraph) -> list[str]:
        errors = []
        present_types = {frag.type for frag in g.nodes.values()}
        for req in REQUIRED_TYPES:
            if req not in present_types:
                errors.append(f"Required type '{req.value}' missing from graph")
        return errors

    def _check_reachability(self, g: PPGraph) -> list[str]:
        errors = []
        if not g.source_ids:
            return ["No source nodes (nodes with in-degree 0)"]
        if not g.terminal_ids:
            return ["No terminal nodes (nodes with out-degree 0)"]

        # BFS from sources — every node must be reachable
        reachable: set[str] = set()
        queue = list(g.source_ids)
        while queue:
            cur = queue.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            queue.extend(g.successors(cur))

        unreachable = set(g.nodes) - reachable
        for nid in unreachable:
            errors.append(f"Node '{nid}' ({g.nodes[nid].type.value}) unreachable from sources")
        return errors

    def _check_no_isolated(self, g: PPGraph) -> list[str]:
        errors = []
        for nid in g.nodes:
            if nid in g.source_ids:
                continue
            if not g.predecessors(nid):
                errors.append(f"Node '{nid}' ({g.nodes[nid].type.value}) has no predecessors and is not a source")
        return errors

    def _check_path_type_coverage(self, g: PPGraph) -> list[str]:
        """Every source-to-terminal path must include all REQUIRED_TYPES."""
        errors = []
        for path in g.all_paths():
            types_in_path = {g.nodes[nid].type for nid in path}
            missing = REQUIRED_TYPES - types_in_path
            if missing:
                missing_names = ", ".join(t.value for t in missing)
                errors.append(
                    f"Path {[g.nodes[n].type.value for n in path]} missing required types: {missing_names}"
                )
        return errors


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------

class PPGraphBuilder:
    """
    Fluent API for constructing a PPGraph programmatically.

    Example:
        g = (PPGraphBuilder()
             .add_fragment(FragmentType.TASK_FRAMING, "Solve: {input}")
             .add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
             .add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer: ")
             .connect("frag-0", "frag-1")
             .connect("frag-1", "frag-2")
             .build())
    """

    def __init__(self):
        self._nodes: dict[str, PromptFragment] = {}
        self._edges: dict[tuple[str, str], Guard] = {}
        self._order: list[str] = []   # insertion order for auto-connect

    def add_fragment(self, ftype: FragmentType, template: str, **metadata) -> "PPGraphBuilder":
        frag = PromptFragment.create(ftype, template, **metadata)
        self._nodes[frag.id] = frag
        self._order.append(frag.id)
        return self

    def add_fragment_obj(self, frag: PromptFragment) -> "PPGraphBuilder":
        self._nodes[frag.id] = frag
        self._order.append(frag.id)
        return self

    def connect(self, src_id: str, dst_id: str,
                guard: Optional[Guard] = None) -> "PPGraphBuilder":
        self._edges[(src_id, dst_id)] = guard or Guard.all_pass()
        return self

    def connect_chain(self, *node_ids: str,
                      guard: Optional[Guard] = None) -> "PPGraphBuilder":
        """Connect a sequence of nodes in order."""
        for src, dst in zip(node_ids, node_ids[1:]):
            self.connect(src, dst, guard)
        return self

    def build(self) -> PPGraph:
        all_srcs = {src for src, _ in self._edges}
        all_dsts = {dst for _, dst in self._edges}

        source_ids = frozenset(
            nid for nid in self._nodes
            if nid not in all_dsts
        )
        terminal_ids = frozenset(
            nid for nid in self._nodes
            if nid not in all_srcs
        )

        g = PPGraph(
            nodes=self._nodes,
            edges=self._edges,
            source_ids=source_ids,
            terminal_ids=terminal_ids,
        )
        GraphValidator().validate_or_raise(g)
        return g

    def node_ids(self) -> list[str]:
        return self._order.copy()
