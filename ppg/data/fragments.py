"""
Seed fragment library for PPG experiments.

Contains curated prompt fragments for the 4 core benchmarks:
    GSM8K    — grade-school math (multi-step arithmetic)
    IFEval   — instruction following with explicit format constraints
    HotpotQA — multi-hop reading comprehension
    MBPP     — Python function synthesis

Fragment design principles
--------------------------
- TASK_FRAMING    : frame the problem clearly; always contains {input}
- REASONING_STYLE : elicit a reasoning strategy before the final answer
- OUTPUT_CONTRACT : constrain output format / length / structure
- COMPRESSION     : optional brevity pressure (useful when token cost λ_c > 0)
- DOMAIN_PRIMER   : optional background knowledge injection for the domain

Graph topology per benchmark
-----------------------------
Each benchmark gets a "rich" graph (full set of optional nodes + branch) and
a "lean" graph (TASK_FRAMING → REASONING_STYLE → OUTPUT_CONTRACT chain).
Use build_graph(benchmark, topology) to construct the PPGraph directly.

Provenance
----------
Fragments are informed by:
    - Wei et al. 2022 (Chain-of-Thought Prompting)
    - Kojima et al. 2022 (Zero-Shot CoT "Let's think step by step")
    - Zhou et al. 2023 (Large Language Models are Human-Level Prompt Engineers)
    - Chen et al. 2022 (Least-to-Most Prompting)
    - GSM8K paper (Cobbe et al. 2021) — #### answer format convention
    - IFEval paper (Zhou et al. 2023) — constraint-compliance framing
    - HotpotQA paper (Yang et al. 2018) — multi-hop supporting fact focus
    - MBPP paper (Austin et al. 2021) — functional test specification style
"""

from __future__ import annotations

from typing import Literal

from ppg.core.graph import FragmentType, PPGraph, PPGraphBuilder, PromptFragment


# ---------------------------------------------------------------------------
# Raw fragment content
# ---------------------------------------------------------------------------

# Each dict maps fragment_type → list of template strings.
# Multiple templates per type → alternative variants.
# Only the first is used in lean graphs; all are available in rich graphs.

FRAGMENTS: dict[str, dict[str, list[str]]] = {

    # -----------------------------------------------------------------------
    # GSM8K — grade-school math
    # -----------------------------------------------------------------------
    "gsm8k": {
        "task_framing": [
            # Primary: direct, minimal framing
            "Solve the following math problem.\n\nProblem: {input}",
            # Variant: explicit "step by step" in the framing itself
            "Solve the following math problem step by step.\n\nProblem: {input}",
        ],
        "domain_primer": [
            # Inject arithmetic conventions before the problem
            (
                "You are solving grade-school math problems. "
                "Use basic arithmetic: addition, subtraction, multiplication, division. "
                "Fractions and percentages may appear. Show intermediate results."
            ),
        ],
        "reasoning_style": [
            # Primary: structured CoT (Wei et al. 2022)
            (
                "Work through the problem step by step.\n"
                "For each step, write the equation and the result.\n"
                "Label each step (Step 1, Step 2, ...)."
            ),
            # Variant: Least-to-Most decomposition (Chen et al. 2022)
            (
                "First, restate what the problem is asking.\n"
                "Then identify the sub-problems you need to solve first.\n"
                "Solve each sub-problem in order, building toward the final answer."
            ),
            # Variant: Zero-shot CoT trigger (Kojima et al. 2022)
            "Let's think step by step.",
        ],
        "compression": [
            # Used when λ_c token penalty is high
            "Keep calculations concise. Skip restating known facts.",
        ],
        "output_contract": [
            # Primary: GSM8K standard — #### separator (Cobbe et al. 2021)
            (
                "After your reasoning, write your final numeric answer on a new line "
                "in this exact format:\n#### [number]\n"
                "Do not include units or explanations after the ####."
            ),
            # Variant: boxed answer style
            "State your final answer as a single number. Write it last, on its own line.",
        ],
    },

    # -----------------------------------------------------------------------
    # IFEval — instruction following
    # -----------------------------------------------------------------------
    "ifeval": {
        "task_framing": [
            # v1: direct compliance framing — minimal preamble, just follow
            (
                "Complete the following task. "
                "Read ALL instructions carefully before writing your response.\n\n"
                "{input}"
            ),
            # v2: explicit constraint decomposition before responding
            (
                "You will respond to the prompt below. "
                "First identify every explicit constraint (format, length, style, content). "
                "Then write a response that satisfies ALL of them.\n\n"
                "{input}"
            ),
            # v3: strict compliance mode — treat each requirement as non-negotiable
            (
                "Follow every instruction below exactly as stated. "
                "Treat each requirement as a hard constraint that cannot be omitted or approximated.\n\n"
                "{input}"
            ),
            # v4: constraint-type tagging — classify then satisfy
            (
                "Read the prompt below and tag each constraint by type "
                "(keyword / format / length / tone / language / structure). "
                "Then write a response satisfying every tagged constraint.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are an instruction-following assistant. "
                "Your task is to comply precisely with every stated constraint. "
                "Constraints may include: word count, formatting (bullet points, numbered lists, "
                "headers, JSON), language, tone, inclusion or exclusion of specific content, "
                "and structural requirements."
            ),
        ],
        "reasoning_style": [
            # v1: numbered checklist audit before writing
            (
                "Before responding, list each constraint you identified, numbered:\n"
                "1. [constraint]\n2. [constraint]\n...\n\n"
                "Then write your response ensuring each constraint is satisfied."
            ),
            # v2: silent planning — no visible scratchpad, just produce the response
            (
                "Carefully identify every constraint in the instructions. "
                "Plan your response to satisfy each one. "
                "Then write the response."
            ),
            # v3: write first, then self-verify and correct
            (
                "Write your response. "
                "Then re-read every constraint and verify your response meets each one. "
                "If any constraint is violated, rewrite the response."
            ),
            # v4: constraint-priority ordering — address hardest constraints first
            (
                "Rank the constraints from hardest to easiest to satisfy. "
                "Build your response starting from the hardest constraint, "
                "then layer in the remaining ones. "
                "Output only the final response."
            ),
        ],
        "compression": [
            "Be concise while still satisfying every constraint. No padding.",
        ],
        "output_contract": [
            # v1: general compliance reminder
            (
                "Your response must satisfy every constraint in the instructions. "
                "Do not include any text outside the requested response format."
            ),
            # v2: self-audit gate before finalizing
            (
                "Before outputting your final response, silently verify: "
                "does it satisfy every stated constraint? "
                "If not, correct it. Output only the compliant response."
            ),
        ],
    },

    # -----------------------------------------------------------------------
    # HotpotQA — multi-hop reading comprehension
    # -----------------------------------------------------------------------
    "hotpotqa": {
        "task_framing": [
            # Primary: open-book with context
            (
                "Answer the question using ONLY the information provided in the context below. "
                "The answer may require connecting facts from multiple passages.\n\n"
                "{input}"
            ),
            # Variant: explicit multi-hop framing
            (
                "Read the context carefully. "
                "The question requires reasoning across two or more passages.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are answering multi-hop questions. "
                "Each question requires synthesizing information from two supporting passages. "
                "Identify the bridge entity — the concept that links the two passages — "
                "and use it to reach the final answer."
            ),
        ],
        "reasoning_style": [
            # Primary: identify supporting facts then derive answer
            (
                "Step 1: Identify which passage(s) contain relevant facts.\n"
                "Step 2: Extract the key facts and how they connect.\n"
                "Step 3: Derive the answer from the connected facts."
            ),
            # Variant: bridge-entity approach
            (
                "Find the bridge entity that connects the two relevant passages. "
                "State what you learn from each passage about this entity. "
                "Then answer the question."
            ),
            # Variant: minimal CoT
            "Think through which facts are needed and how they connect.",
        ],
        "compression": [
            "Reason briefly. Do not quote entire passages — extract only the relevant facts.",
        ],
        "output_contract": [
            # HotpotQA answers are typically short spans
            (
                "Your answer must be a short phrase or a few words — NOT a full sentence. "
                "Do not include any explanation in your answer. "
                "Write only the answer."
            ),
            # Variant: even stricter
            "Answer in 1–5 words. No punctuation unless part of the answer.",
        ],
    },

    # -----------------------------------------------------------------------
    # MBPP — Python function synthesis
    # -----------------------------------------------------------------------
    "mbpp": {
        "task_framing": [
            # Primary: function-only framing (Austin et al. 2021)
            (
                "Write a Python function that solves the following problem.\n\n"
                "{input}\n\n"
                "Provide only the function definition. "
                "Do not include example usage or explanations outside the function."
            ),
            # Variant: docstring-first approach
            (
                "Implement a Python function for the following specification.\n\n"
                "{input}\n\n"
                "Write the function with a concise docstring, then the implementation."
            ),
        ],
        "domain_primer": [
            (
                "You are writing Python 3 code. "
                "Use built-in functions and the standard library when appropriate. "
                "Prefer readability over cleverness. "
                "Handle edge cases: empty inputs, zero values, single elements."
            ),
        ],
        "reasoning_style": [
            # Primary: plan then implement
            (
                "Before writing code:\n"
                "1. Identify the inputs and expected outputs.\n"
                "2. Identify edge cases (empty, None, zero, single element).\n"
                "3. Choose a simple algorithm.\n\n"
                "Then write the function."
            ),
            # Variant: test-driven thinking
            (
                "Think about what the test cases would look like. "
                "Make sure your implementation passes obvious edge cases: "
                "empty input, single element, typical case, large input."
            ),
            # Variant: concise
            "Think about edge cases before writing the function.",
        ],
        "compression": [
            (
                "Write minimal, idiomatic Python. "
                "Avoid unnecessary variables and comments."
            ),
        ],
        "output_contract": [
            # Primary: clean function only
            (
                "Output ONLY the Python function definition. "
                "No markdown fences, no usage examples, no explanations.\n"
                "Start your response with 'def '."
            ),
            # Variant: allow markdown (easier for some models)
            (
                "Output the Python function inside a ```python code block. "
                "Nothing outside the code block."
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Graph topologies
# ---------------------------------------------------------------------------

Topology = Literal["lean", "rich"]


def build_graph(
    benchmark: str,
    topology:  Topology = "rich",
    variant:   int = 0,
) -> PPGraph:
    """
    Construct a PPGraph from the seed fragment library.

    Parameters
    ----------
    benchmark : "gsm8k" | "ifeval" | "hotpotqa" | "mbpp"
    topology  : "lean"  → linear chain (3 nodes, no optional branches)
                "rich"  → extended graph with DOMAIN_PRIMER and COMPRESSION
    variant   : which template variant to use (0 = primary)

    Lean topology
    -------------
    TASK_FRAMING → REASONING_STYLE → OUTPUT_CONTRACT

    Rich topology
    -------------
    DOMAIN_PRIMER → TASK_FRAMING → REASONING_STYLE → OUTPUT_CONTRACT
                                                   ↘ COMPRESSION → OUTPUT_CONTRACT
    (COMPRESSION branch is selected by the bandit when token cost is high)

    Returns
    -------
    Validated PPGraph ready for PPGExecutor.
    """
    if benchmark not in FRAGMENTS:
        raise ValueError(
            f"Unknown benchmark: {benchmark!r}. "
            f"Available: {sorted(FRAGMENTS.keys())}"
        )

    frags = FRAGMENTS[benchmark]
    v = variant

    def pick(key: str) -> str:
        templates = frags.get(key, [])
        if not templates:
            raise KeyError(f"No fragments for key {key!r} in benchmark {benchmark!r}")
        return templates[v % len(templates)]

    b = PPGraphBuilder()

    if topology == "lean":
        return _build_lean(b, pick)
    elif topology == "rich":
        return _build_rich(b, pick, frags)
    else:
        raise ValueError(f"Unknown topology: {topology!r}. Use 'lean' or 'rich'.")


def _build_lean(b: PPGraphBuilder, pick) -> PPGraph:
    b.add_fragment(FragmentType.TASK_FRAMING,    pick("task_framing"))
    b.add_fragment(FragmentType.REASONING_STYLE, pick("reasoning_style"))
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, pick("output_contract"))
    tf, rs, oc = b.node_ids()
    b.connect_chain(tf, rs, oc)
    return b.build()


def _build_rich(b: PPGraphBuilder, pick, frags: dict) -> PPGraph:
    """
    Multi-variant parallel graph — bandit chooses one variant at each level.

    DOMAIN_PRIMER → TF_v0 ─┬→ RS_v0 ─┬→ OC_v0
                   TF_v1 ─┘  RS_v1 ─┤  OC_v1
                              RS_v2 ─┘
                                └→ COMP ─→ OC_*

    All TF variants connect to all RS variants.
    RS variants connect directly to all OC variants and, when present, to
    COMP before OC. Compression is an optional post-reasoning add-on.
    """
    def _add(ftype: FragmentType, template: str) -> str:
        frag = PromptFragment.create(ftype, template)
        b.add_fragment_obj(frag)
        return frag.id

    has_primer      = "domain_primer" in frags
    has_compression = "compression"   in frags

    dp_id   = _add(FragmentType.DOMAIN_PRIMER,    frags["domain_primer"][0]) if has_primer else None
    tf_ids  = [_add(FragmentType.TASK_FRAMING,    t) for t in frags.get("task_framing",    [])]
    rs_ids  = [_add(FragmentType.REASONING_STYLE, t) for t in frags.get("reasoning_style", [])]
    oc_ids  = [_add(FragmentType.OUTPUT_CONTRACT, t) for t in frags.get("output_contract", [])]
    comp_id = _add(FragmentType.COMPRESSION,       frags["compression"][0]) if has_compression else None

    for tf_id in tf_ids:
        if dp_id:
            b.connect(dp_id, tf_id)
        for rs_id in rs_ids:
            b.connect(tf_id, rs_id)

    for rs_id in rs_ids:
        for oc_id in oc_ids:
            b.connect(rs_id, oc_id)
        if comp_id:
            b.connect(rs_id, comp_id)

    if comp_id:
        for oc_id in oc_ids:
            b.connect(comp_id, oc_id)

    return b.build()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def available_benchmarks() -> list[str]:
    return sorted(FRAGMENTS.keys())


def fragment_count(benchmark: str) -> dict[str, int]:
    """Returns {fragment_type: n_variants} for a benchmark."""
    if benchmark not in FRAGMENTS:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    return {k: len(v) for k, v in FRAGMENTS[benchmark].items()}


def list_fragments(benchmark: str, fragment_type: str) -> list[str]:
    """Return all variant templates for a (benchmark, fragment_type) pair."""
    if benchmark not in FRAGMENTS:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    return list(FRAGMENTS[benchmark].get(fragment_type, []))
