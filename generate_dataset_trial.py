#!/usr/bin/env python3
"""
Resynth Dataset Generation Script

Builds prompts by:
1. Loading 3 random seed examples (with replacement) from the HF dataset
2. Formatting them via example_format.txt and inserting into initial.txt
3. Generating 0-3 extra_details with weighted probabilities
4. Applying custom string formatting to each extra detail:
   - (a/b/c) → pick 1-3 options, join by comma
   - [a/b/c] → pick exactly 1 option
5. Printing the final assembled prompt
"""

import os
import re
import random

from datasets import load_dataset

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

#SEED = 64
#random.seed(SEED)

HF_SEED_DATASET = "G-reen/Resynth-Seed"

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "init")
INITIAL_PROMPT_PATH = os.path.join(PROMPTS_DIR, "initial.txt")
EXAMPLE_FORMAT_PATH = os.path.join(PROMPTS_DIR, "example_format.txt")
EXTRA_DETAILS_DIR = os.path.join(PROMPTS_DIR, "extra_details")
EVOLVE_PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "evolve")

# Extra details count distribution
# 0 details: 40%, 1: 30%, 2: 20%, 3: 10%
EXTRA_DETAILS_WEIGHTS = [0.2, 0.4, 0.3, 0.1]
EXTRA_DETAILS_COUNTS = [0, 1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# Load Templates & Seed Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_text(path: str) -> str:
    """Load a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_seed_dataset():
    """Load the seed dataset from HuggingFace and return a list of seed texts."""
    print(f"📦  Loading seed dataset from {HF_SEED_DATASET} ...")
    ds = load_dataset(HF_SEED_DATASET, split="train")
    seeds = [row["seed"] for row in ds]
    print(f"    Loaded {len(seeds):,} seeds.")
    return seeds


def load_extra_detail_templates() -> list[tuple[str, str]]:
    """Load all extra detail templates from the extra_details folder.

    Returns a list of (filename_stem, template_text) tuples.
    """
    templates = []
    for fname in sorted(os.listdir(EXTRA_DETAILS_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(EXTRA_DETAILS_DIR, fname)
        text = load_text(path)
        stem = os.path.splitext(fname)[0]
        templates.append((stem, text))
    return templates


# ═══════════════════════════════════════════════════════════════════════════════
# Custom String Formatter
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_multi_option(match: re.Match) -> str:
    """Handle (a/b/c) — pick 1-3 options, join by comma."""
    inner = match.group(1)
    options = [opt.strip() for opt in inner.split("/")]
    k = random.randint(1, min(3, len(options)))
    chosen = random.sample(options, k)
    return ", ".join(chosen)


def _resolve_single_option(match: re.Match) -> str:
    """Handle [a/b/c] — pick exactly 1 option."""
    inner = match.group(1)
    options = [opt.strip() for opt in inner.split("/")]
    return random.choice(options)


def format_detail(template: str) -> str:
    """Apply the custom string formatting rules to an extra detail template.

    - (a/b/c)  → randomly pick 1-3 of the options, join with ', '
    - [a/b/c]  → randomly pick exactly 1 of the options
    """
    # Process parenthesis groups first, then brackets
    # Use a non-greedy match to handle multiple groups on the same line
    result = re.sub(r"\(([^)]+)\)", _resolve_multi_option, template)
    result = re.sub(r"\[([^\]]+)\]", _resolve_single_option, result)
    return result


def get_evolve_prompt() -> str:
    """Load a random evolve prompt and append instruction."""
    if not os.path.exists(EVOLVE_PROMPTS_DIR):
        return "Error: Evolve prompts directory not found."
        
    files = [f for f in os.listdir(EVOLVE_PROMPTS_DIR) if f.endswith(".txt")]
    if not files:
        return "Error: No evolve prompts found."
        
    chosen_file = random.choice(files)
    content = load_text(os.path.join(EVOLVE_PROMPTS_DIR, chosen_file))
    return f"{content}\n\nBegin your output with ### Prompt:."


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    seeds: list[str],
    initial_template: str,
    example_format_template: str,
    extra_detail_templates: list[tuple[str, str]],
) -> str:
    """Build a single prompt.

    1. Pick 1-3 seeds with replacement → format via example_format.txt
    2. Substitute into initial.txt at {{examples}}
    3. Pick 0-3 extra details, format each, substitute at {{extra_details}}
    """
    # ── Step 1: Pick 1-3 seeds with replacement ──
    num_seeds = random.randint(1, 3)
    chosen_seeds = random.choices(seeds, k=num_seeds)

    # ── Step 2: Format each seed as an example block ──
    example_blocks = []
    for i, seed_text in enumerate(chosen_seeds, start=1):
        block = example_format_template.replace("{{n}}", str(i))
        block = block.replace("{{example}}", seed_text)
        example_blocks.append(block)

    examples_str = "\n\n".join(example_blocks)

    # ── Step 3: Determine number of extra details ──
    num_extra = random.choices(EXTRA_DETAILS_COUNTS, weights=EXTRA_DETAILS_WEIGHTS, k=1)[0]

    extra_details_str = ""
    if num_extra > 0 and extra_detail_templates:
        # Randomly pick `num_extra` detail templates (without replacement)
        # Using sample ensures we don't pick the same category (e.g. complexity) twice
        chosen_details = random.sample(extra_detail_templates, k=num_extra)
        formatted = [format_detail(template) for _stem, template in chosen_details]
        extra_details_str = "\n" + " ".join(formatted)

    # ── Step 4: Assemble the final prompt ──
    prompt = initial_template.replace("{{examples}}", examples_str)
    prompt = prompt.replace("{{extra_details}}", extra_details_str)

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Resynth Prompt Generator — Test Mode")
    print("=" * 72)

    # Load resources
    initial_template = load_text(INITIAL_PROMPT_PATH)
    example_format_template = load_text(EXAMPLE_FORMAT_PATH)
    extra_detail_templates = load_extra_detail_templates()
    seeds = load_seed_dataset()

    print(f"\n  Loaded {len(extra_detail_templates)} extra detail templates:")
    for stem, _ in extra_detail_templates:
        print(f"    • {stem}")
    print()

    # Generate and print a single prompt for testing
    prompt = build_prompt(seeds, initial_template, example_format_template, extra_detail_templates)

    print("=" * 72)
    print("  GENERATED PROMPT")
    print("=" * 72)
    print(prompt)
    print("=" * 72)

    evolve_prompt = get_evolve_prompt()
    print("\n" + "=" * 72)
    print("  EVOLVE PROMPT SAMPLE")
    print("=" * 72)
    print(evolve_prompt)
    print("=" * 72)


if __name__ == "__main__":
    main()
