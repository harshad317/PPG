"""
Seed fragment library for PPG experiments.

Contains curated prompt fragments for the 8 core benchmarks:
    GSM8K         — grade-school math (multi-step arithmetic)
    IFBench       — instruction following with constraint satisfaction
    TruthfulQA    — factual accuracy under adversarial framing
    BIG-Bench Hard— challenging multi-step reasoning
    ARC-Challenge — science multiple-choice questions
    LiveBench-Math— competition-level math problems
    HotpotQA      — multi-hop reading comprehension
    MBPP          — Python function synthesis

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
        "few_shot": [
            # v1: multi-step arithmetic with clear step labels
            (
                "Here is an example of solving a math problem step by step:\n\n"
                "Problem: A store sells apples for $2 each. Sarah buys 5 apples "
                "and pays with a $20 bill. How much change does she get?\n"
                "Solution:\n"
                "Step 1: Cost of apples = 5 * $2 = $10\n"
                "Step 2: Change = $20 - $10 = $10\n"
                "#### 10\n\n"
                "Now solve the problem above with the same step-by-step approach."
            ),
            # v2: multi-step with intermediate quantities
            (
                "Here is an example of solving a math problem step by step:\n\n"
                "Problem: Tom reads 3 pages per minute. He reads for 2 hours. "
                "How many pages does he read?\n"
                "Solution:\n"
                "Step 1: Convert hours to minutes: 2 * 60 = 120 minutes\n"
                "Step 2: Total pages = 120 * 3 = 360\n"
                "#### 360\n\n"
                "Now solve the problem above with the same step-by-step approach."
            ),
            # v3: percentage/fraction example
            (
                "Here is an example of solving a math problem step by step:\n\n"
                "Problem: A shirt costs $40. It is on sale for 25% off. "
                "What is the sale price?\n"
                "Solution:\n"
                "Step 1: Discount amount = 40 * 25/100 = $10\n"
                "Step 2: Sale price = $40 - $10 = $30\n"
                "#### 30\n\n"
                "Now solve the problem above with the same step-by-step approach."
            ),
            # v4: control variant
            "Solve the problem step by step and give the final numeric answer.",
        ],
    },

    # -----------------------------------------------------------------------
    # IFBench — instruction following with constraint satisfaction
    # -----------------------------------------------------------------------
    "ifbench": {
        "task_framing": [
            # v1: direct compliance framing
            (
                "Complete the following task. "
                "Read ALL instructions carefully before writing your response.\n\n"
                "{input}"
            ),
            # v2: explicit constraint decomposition before responding
            (
                "You will respond to the prompt below. "
                "Silently identify every explicit constraint "
                "(format, length, style, content). "
                "Then write only the response that satisfies ALL of them.\n\n"
                "{input}"
            ),
            # v3: strict compliance mode
            (
                "Follow every instruction below exactly as stated. "
                "Treat each requirement as a hard constraint that cannot be omitted or approximated.\n\n"
                "{input}"
            ),
            # v4: constraint-type tagging
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
            (
                "You are a constraint-satisfaction engine. "
                "Every instruction contains explicit requirements that must be met exactly. "
                "Never approximate a constraint — if it says 50 words, write exactly 50. "
                "If it says JSON, output valid JSON with no surrounding text. "
                "Precision over creativity."
            ),
        ],
        "reasoning_style": [
            # v1: numbered checklist audit before writing
            (
                "Before responding, silently check each constraint one by one. "
                "Use that checklist only to guide the final answer; do not output "
                "the checklist or any analysis."
            ),
            # v2: silent planning
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
            # v4: constraint-priority ordering
            (
                "Rank the constraints from hardest to easiest to satisfy. "
                "Build your response starting from the hardest constraint, "
                "then layer in the remaining ones. "
                "Output only the final response."
            ),
            # v5: explicit constraint extraction
            (
                "First, silently extract every constraint from the instructions "
                "into a numbered list (format, length, keywords, tone, structure). "
                "Then compose a response that satisfies each numbered constraint. "
                "Output only the final response, not the list."
            ),
            # v6: count-then-write
            (
                "If the instructions mention a specific count "
                "(words, sentences, paragraphs, items), determine the exact "
                "target number first. Write your response to hit that number "
                "precisely. Count again to verify before finalizing."
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
            # v3: length-aware enforcement
            (
                "If the instructions specify a word count, sentence count, or "
                "paragraph count, count carefully before finalizing. "
                "Rewrite if any length constraint is violated. "
                "Output only the final compliant response."
            ),
            # v4: format-strict enforcement
            (
                "If the instructions require a specific format "
                "(numbered list, bullet points, JSON, sections with headers, "
                "all caps, lowercase, or markdown), "
                "produce output in exactly that format. "
                "Do not add any extra text, preamble, or commentary."
            ),
            # v5: keyword/content inclusion gate
            (
                "If the instructions require including or excluding specific "
                "words, phrases, or content, verify each requirement is met. "
                "Do not omit required keywords. "
                "Do not include forbidden content. "
                "Output only the compliant response."
            ),
            # v6: complete-response guarantee
            (
                "Your response must be complete — do not cut off mid-sentence "
                "or leave any part of the task unfinished. "
                "Verify every constraint is satisfied in the full response "
                "before outputting it."
            ),
        ],
        "few_shot": [
            # v1: word count constraint example
            (
                "Here is an example of correctly following constraints:\n\n"
                "Instruction: Write a sentence about the ocean. "
                "Your response must contain exactly 12 words.\n"
                "Response: The vast ocean stretches endlessly, its blue waves "
                "crashing against the shore.\n\n"
                "Now follow ALL constraints in the task above with the same precision."
            ),
            # v2: format constraint example
            (
                "Here is an example of correctly following constraints:\n\n"
                "Instruction: List three colors. Use a numbered list. "
                "Each item must be one word.\n"
                "Response:\n1. Red\n2. Blue\n3. Green\n\n"
                "Now follow ALL constraints in the task above with the same precision."
            ),
            # v3: keyword inclusion + format example
            (
                "Here is an example of correctly following constraints:\n\n"
                "Instruction: Write a paragraph about dogs. "
                "Include the words 'loyal' and 'companion'. "
                "Use exactly 2 sentences.\n"
                "Response: Dogs are loyal animals that have been by our side "
                "for thousands of years. Every dog owner knows the joy of having "
                "such a devoted companion.\n\n"
                "Now follow ALL constraints in the task above with the same precision."
            ),
            # v4: control variant
            "Follow the constraints in the task directly.",
        ],
    },

    # -----------------------------------------------------------------------
    # TruthfulQA — factual accuracy under adversarial framing
    # -----------------------------------------------------------------------
    "truthfulqa": {
        "task_framing": [
            # v1: direct truthfulness framing
            (
                "Answer the following question truthfully. "
                "If the question contains a false premise, say so.\n\n"
                "{input}"
            ),
            # v2: common-misconception awareness
            (
                "Answer the question below. Be careful — many similar questions "
                "have popular but incorrect answers. Prioritize accuracy over "
                "what sounds right.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are an expert fact-checker. "
                "Many questions are designed to elicit common misconceptions. "
                "Ground your answer in verifiable facts. "
                "If unsure, say you are uncertain rather than guessing."
            ),
        ],
        "reasoning_style": [
            # v1: consider-then-answer
            (
                "Before answering, consider whether this question contains "
                "a common misconception or false assumption.\n"
                "If it does, address the misconception directly.\n"
                "Then give your truthful answer."
            ),
            # v2: evidence-based
            (
                "Think about what evidence supports the correct answer. "
                "Distinguish between popular belief and verified fact. "
                "Then answer."
            ),
            # v3: minimal CoT
            "Think carefully about whether common assumptions apply here.",
        ],
        "compression": [
            "Answer directly. No unnecessary hedging or caveats.",
        ],
        "output_contract": [
            # v1: short factual answer
            (
                "Give a clear, concise, truthful answer. "
                "If the premise is false, explain why briefly. "
                "Keep your answer to 1-3 sentences."
            ),
            # v2: answer-only
            "Write only your answer. No preamble, no explanation.",
        ],
        "few_shot": [
            # v1: misconception correction example
            (
                "Here is an example of answering truthfully:\n\n"
                "Question: What happens if you swallow gum?\n"
                "Answer: Swallowed gum passes through the digestive system "
                "and is excreted normally. It is not true that it stays in "
                "your stomach for seven years.\n\n"
                "Now answer the question above with the same commitment to accuracy."
            ),
            # v2: false premise example
            (
                "Here is an example of answering truthfully:\n\n"
                "Question: Which country has the most pyramids?\n"
                "Answer: Sudan has the most pyramids, not Egypt as commonly "
                "believed. Sudan has over 200 pyramids.\n\n"
                "Now answer the question above with the same commitment to accuracy."
            ),
            # v3: nuanced answer example
            (
                "Here is an example of answering truthfully:\n\n"
                "Question: Do we only use 10% of our brains?\n"
                "Answer: No, this is a myth. Brain imaging shows that all "
                "areas of the brain have a function and are active over the "
                "course of a day.\n\n"
                "Now answer the question above with the same commitment to accuracy."
            ),
            # v4: control variant
            "Answer the question truthfully and accurately.",
        ],
    },

    # -----------------------------------------------------------------------
    # BIG-Bench Hard — challenging multi-step reasoning
    # -----------------------------------------------------------------------
    "bigbench_hard": {
        "task_framing": [
            # v1: direct reasoning framing
            (
                "Solve the following reasoning problem carefully.\n\n"
                "{input}"
            ),
            # v2: explicit multi-step framing
            (
                "The following problem requires careful multi-step reasoning. "
                "Read the problem fully before beginning to solve it.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are solving challenging reasoning tasks that require "
                "careful logical thinking. These problems often have "
                "counter-intuitive answers — do not rely on surface-level "
                "pattern matching. Trace the logic step by step."
            ),
        ],
        "reasoning_style": [
            # v1: structured step-by-step
            (
                "Break the problem into logical steps.\n"
                "For each step, state what you know and what you can deduce.\n"
                "Only after completing all steps, state your final answer."
            ),
            # v2: enumerate-then-eliminate
            (
                "First, identify the possible answers or outcomes.\n"
                "Then systematically evaluate each one against the given constraints.\n"
                "Eliminate options that fail any constraint. "
                "Select the remaining valid answer."
            ),
            # v3: minimal CoT
            "Think through this step by step, being careful with each logical inference.",
        ],
        "compression": [
            "Reason concisely. State conclusions, skip obvious intermediate steps.",
        ],
        "output_contract": [
            # v1: labeled final answer
            (
                "After your reasoning, write your final answer on a new line "
                "in this exact format:\nAnswer: [your answer]\n"
                "Give only the answer value, no extra explanation after it."
            ),
            # v2: answer-only
            "State your final answer clearly. Write it last, on its own line.",
        ],
        "few_shot": [
            # v1: logical deduction example
            (
                "Here is an example of careful step-by-step reasoning:\n\n"
                "Problem: If all roses are flowers and some flowers fade quickly, "
                "can we conclude that some roses fade quickly?\n"
                "Reasoning: All roses are flowers (given). Some flowers fade "
                "quickly (given). But 'some flowers' may not include any roses. "
                "We cannot conclude that some roses fade quickly.\n"
                "Answer: No\n\n"
                "Now solve the problem above with the same careful reasoning."
            ),
            # v2: ordering/sequencing example
            (
                "Here is an example of careful step-by-step reasoning:\n\n"
                "Problem: Alice is taller than Bob. Carol is shorter than Bob. "
                "Who is the shortest?\n"
                "Reasoning: Alice > Bob (given). Bob > Carol (given). "
                "So Alice > Bob > Carol. Carol is shortest.\n"
                "Answer: Carol\n\n"
                "Now solve the problem above with the same careful reasoning."
            ),
            # v3: causal reasoning example
            (
                "Here is an example of careful step-by-step reasoning:\n\n"
                "Problem: A bat and a ball cost $1.10 total. The bat costs "
                "$1.00 more than the ball. How much does the ball cost?\n"
                "Reasoning: Let ball = x. Bat = x + 1.00. "
                "x + (x + 1.00) = 1.10. 2x = 0.10. x = 0.05.\n"
                "Answer: $0.05\n\n"
                "Now solve the problem above with the same careful reasoning."
            ),
            # v4: control variant
            "Solve the reasoning problem step by step.",
        ],
    },

    # -----------------------------------------------------------------------
    # ARC-Challenge — science multiple-choice questions
    # -----------------------------------------------------------------------
    "arc_challenge": {
        "task_framing": [
            # v1: MCQ framing with answer label
            (
                "Answer the following multiple-choice science question. "
                "Choose the best answer.\n\n"
                "{input}"
            ),
            # v2: reasoning-first MCQ framing
            (
                "Read the following science question and its answer choices. "
                "Reason about which answer is correct before choosing.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are answering grade-school and middle-school science questions. "
                "Topics include physics, chemistry, biology, and earth science. "
                "Apply scientific principles and common knowledge to select the "
                "correct answer."
            ),
        ],
        "reasoning_style": [
            # v1: eliminate wrong answers
            (
                "Consider each answer choice.\n"
                "Eliminate choices that contradict known scientific principles.\n"
                "Select the remaining correct answer."
            ),
            # v2: recall relevant principle then apply
            (
                "First, identify which scientific concept or principle is being tested.\n"
                "Then apply that principle to determine the correct answer."
            ),
            # v3: minimal CoT
            "Think about which scientific principle applies, then choose the best answer.",
        ],
        "compression": [
            "Choose the correct answer. Brief reasoning only.",
        ],
        "output_contract": [
            # v1: letter answer format
            (
                "After your reasoning, write your final answer on a new line "
                "as just the answer label (A, B, C, or D).\n"
                "Do not repeat the answer text."
            ),
            # v2: answer-only
            "State only the letter of the correct answer (A, B, C, or D).",
        ],
        "few_shot": [
            # v1: physics example
            (
                "Here is an example of answering a science question:\n\n"
                "Question: What force keeps the planets in orbit around the Sun?\n"
                "A) Magnetism B) Friction C) Gravity D) Inertia\n"
                "Reasoning: Gravity is the force that attracts objects with mass "
                "toward each other. It keeps planets in orbit around the Sun.\n"
                "Answer: C\n\n"
                "Now answer the question above using the same approach."
            ),
            # v2: biology example
            (
                "Here is an example of answering a science question:\n\n"
                "Question: What is the function of chlorophyll in plants?\n"
                "A) Absorb water B) Absorb light for photosynthesis "
                "C) Transport nutrients D) Store energy\n"
                "Reasoning: Chlorophyll is the pigment in plant cells that "
                "absorbs light energy used in photosynthesis.\n"
                "Answer: B\n\n"
                "Now answer the question above using the same approach."
            ),
            # v3: earth science example
            (
                "Here is an example of answering a science question:\n\n"
                "Question: Which layer of Earth is the thinnest?\n"
                "A) Inner core B) Outer core C) Mantle D) Crust\n"
                "Reasoning: The crust is the outermost and thinnest layer of Earth, "
                "ranging from 5 to 70 km thick.\n"
                "Answer: D\n\n"
                "Now answer the question above using the same approach."
            ),
            # v4: control variant
            "Choose the correct answer for the science question.",
        ],
    },

    # -----------------------------------------------------------------------
    # LiveBench-Math — competition-level math problems
    # -----------------------------------------------------------------------
    "livebench_math": {
        "task_framing": [
            # v1: direct math framing
            (
                "Solve the following math problem. Show your work.\n\n"
                "{input}"
            ),
            # v2: competition-style framing
            (
                "Solve the following competition math problem. "
                "Think carefully — these problems often require insight "
                "beyond straightforward calculation.\n\n"
                "{input}"
            ),
        ],
        "domain_primer": [
            (
                "You are solving competition-level math problems. "
                "These may involve algebra, number theory, combinatorics, "
                "geometry, or probability. Look for patterns, use algebraic "
                "manipulation, and verify your answer."
            ),
        ],
        "reasoning_style": [
            # v1: structured solution
            (
                "Work through the problem systematically:\n"
                "1. Identify what is being asked.\n"
                "2. Determine the mathematical approach.\n"
                "3. Execute the computation step by step.\n"
                "4. Verify your answer."
            ),
            # v2: explore-then-solve
            (
                "Consider different approaches to the problem. "
                "Try the most promising approach. "
                "If stuck, try another. "
                "Verify the final answer by substitution or estimation."
            ),
            # v3: minimal CoT
            "Solve step by step. Check your answer at the end.",
        ],
        "compression": [
            "Solve concisely. Show key steps only, skip trivial algebra.",
        ],
        "output_contract": [
            # v1: #### answer format (consistent with gsm8k)
            (
                "After your solution, write your final numeric answer on a new line "
                "in this exact format:\n#### [number]\n"
                "Do not include units or explanations after the ####."
            ),
            # v2: answer-only
            "State your final answer as a single number on its own line.",
        ],
        "few_shot": [
            # v1: algebraic manipulation example
            (
                "Here is an example of solving a competition math problem:\n\n"
                "Problem: If x + 1/x = 5, find x^2 + 1/x^2.\n"
                "Solution:\n"
                "Step 1: Square both sides: (x + 1/x)^2 = 25\n"
                "Step 2: Expand: x^2 + 2 + 1/x^2 = 25\n"
                "Step 3: Subtract 2: x^2 + 1/x^2 = 23\n"
                "#### 23\n\n"
                "Now solve the problem above with the same rigor."
            ),
            # v2: counting/combinatorics example
            (
                "Here is an example of solving a competition math problem:\n\n"
                "Problem: How many ways can you arrange the letters in BOOK?\n"
                "Solution:\n"
                "Step 1: BOOK has 4 letters with O repeated twice.\n"
                "Step 2: Arrangements = 4! / 2! = 24 / 2 = 12\n"
                "#### 12\n\n"
                "Now solve the problem above with the same rigor."
            ),
            # v3: number theory example
            (
                "Here is an example of solving a competition math problem:\n\n"
                "Problem: What is the remainder when 7^100 is divided by 5?\n"
                "Solution:\n"
                "Step 1: 7 mod 5 = 2. So find 2^100 mod 5.\n"
                "Step 2: Powers of 2 mod 5 cycle: 2,4,3,1,2,4,3,1... period 4.\n"
                "Step 3: 100 mod 4 = 0, so 2^100 mod 5 = 1.\n"
                "#### 1\n\n"
                "Now solve the problem above with the same rigor."
            ),
            # v4: control variant
            "Solve the math problem and give the final numeric answer.",
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
        "few_shot": [
            # v1: bridge-entity multi-hop example
            (
                "Here is an example of multi-hop reasoning:\n\n"
                "Context: Passage 1: The Eiffel Tower is located in Paris. "
                "Passage 2: Paris is the capital of France.\n"
                "Question: In which country is the Eiffel Tower located?\n"
                "Reasoning: Eiffel Tower is in Paris (Passage 1). "
                "Paris is in France (Passage 2).\n"
                "Answer: France\n\n"
                "Now answer the question above using the same multi-hop approach."
            ),
            # v2: comparison-type multi-hop example
            (
                "Here is an example of multi-hop reasoning:\n\n"
                "Context: Passage 1: Alice was born in 1990. "
                "Passage 2: Bob was born in 1985.\n"
                "Question: Who is older, Alice or Bob?\n"
                "Reasoning: Alice born 1990 (Passage 1), Bob born 1985 (Passage 2). "
                "1985 < 1990, so Bob is older.\n"
                "Answer: Bob\n\n"
                "Now answer the question above using the same multi-hop approach."
            ),
            # v3: entity-attribute lookup example
            (
                "Here is an example of multi-hop reasoning:\n\n"
                "Context: Passage 1: The director of Inception is Christopher Nolan. "
                "Passage 2: Christopher Nolan was born in London.\n"
                "Question: Where was the director of Inception born?\n"
                "Reasoning: Director of Inception = Christopher Nolan (Passage 1). "
                "Nolan born in London (Passage 2).\n"
                "Answer: London\n\n"
                "Now answer the question above using the same multi-hop approach."
            ),
            # v4: control variant
            "Answer the question using facts from the provided context.",
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
        "few_shot": [
            # v1: simple list processing example
            (
                "Here is an example of writing a clean Python function:\n\n"
                "Task: Write a function to find the maximum element in a list.\n"
                "Solution:\n"
                "def find_max(lst):\n"
                "    if not lst:\n"
                "        return None\n"
                "    return max(lst)\n\n"
                "Now write the function for the task above with the same clarity."
            ),
            # v2: string processing example
            (
                "Here is an example of writing a clean Python function:\n\n"
                "Task: Write a function that checks if a string is a palindrome.\n"
                "Solution:\n"
                "def is_palindrome(s):\n"
                "    s = s.lower().strip()\n"
                "    return s == s[::-1]\n\n"
                "Now write the function for the task above with the same clarity."
            ),
            # v3: numeric/math example with edge case handling
            (
                "Here is an example of writing a clean Python function:\n\n"
                "Task: Write a function to compute the factorial of a number.\n"
                "Solution:\n"
                "def factorial(n):\n"
                "    if n <= 1:\n"
                "        return 1\n"
                "    return n * factorial(n - 1)\n\n"
                "Now write the function for the task above with the same clarity."
            ),
            # v4: control variant
            "Write a correct Python function for the task.",
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
    include_few_shot: bool = False,
) -> PPGraph:
    """
    Construct a PPGraph from the seed fragment library.

    Parameters
    ----------
    benchmark : "gsm8k" | "ifbench" | "truthfulqa" | "bigbench_hard" | "arc_challenge" | "livebench_math" | "hotpotqa" | "mbpp"
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
        return _build_rich(b, pick, frags, include_few_shot=include_few_shot)
    else:
        raise ValueError(f"Unknown topology: {topology!r}. Use 'lean' or 'rich'.")


def _build_lean(b: PPGraphBuilder, pick) -> PPGraph:
    b.add_fragment(FragmentType.TASK_FRAMING,    pick("task_framing"))
    b.add_fragment(FragmentType.REASONING_STYLE, pick("reasoning_style"))
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, pick("output_contract"))
    tf, rs, oc = b.node_ids()
    b.connect_chain(tf, rs, oc)
    return b.build()


def _build_rich(b: PPGraphBuilder, pick, frags: dict, include_few_shot: bool = False) -> PPGraph:
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
    has_few_shot    = include_few_shot and "few_shot" in frags

    dp_id   = _add(FragmentType.DOMAIN_PRIMER,    frags["domain_primer"][0]) if has_primer else None
    tf_ids  = [_add(FragmentType.TASK_FRAMING,    t) for t in frags.get("task_framing",    [])]
    fs_ids  = [_add(FragmentType.FEW_SHOT,        t) for t in frags.get("few_shot",        [])] if has_few_shot else []
    rs_ids  = [_add(FragmentType.REASONING_STYLE, t) for t in frags.get("reasoning_style", [])]
    oc_ids  = [_add(FragmentType.OUTPUT_CONTRACT, t) for t in frags.get("output_contract", [])]
    comp_id = _add(FragmentType.COMPRESSION,       frags["compression"][0]) if has_compression else None

    # DP → TF
    for tf_id in tf_ids:
        if dp_id:
            b.connect(dp_id, tf_id)

    if fs_ids:
        # TF → FS → RS (few-shot between task framing and reasoning)
        for tf_id in tf_ids:
            for fs_id in fs_ids:
                b.connect(tf_id, fs_id)
        for fs_id in fs_ids:
            for rs_id in rs_ids:
                b.connect(fs_id, rs_id)
    else:
        # TF → RS (no few-shot layer)
        for tf_id in tf_ids:
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
