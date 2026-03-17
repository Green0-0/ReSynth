#!/usr/bin/env python3
"""
Resynth Dataset Generation Script (Batched & Evolved via vLLM)

Builds initial prompts, feeds them into StepFun NVFP4 via vLLM, 
and sequentially evolves each conversation 1-3 times in static batches.
Exports the final result to a JSONL and pushes to the Hugging Face Hub.
"""

import os
import re
import random
import json
from tqdm import tqdm

from datasets import load_dataset, Dataset
from huggingface_hub import login
from vllm import LLM, SamplingParams

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Generation Targets
TARGET_TOTAL_PROMPTS = 160      # Stop after generating this many final prompts
BATCH_SIZE = 16                 # Number of concurrent conversations per chunk
HF_OUTPUT_REPO = "G-reen/Resynth-Base" # REPLACE WITH YOUR REPO

# Model / vLLM Config
MODEL_ID = "stepfun-ai/Step-3.5-Flash-FP8"
TENSOR_PARALLEL_SIZE = 2
MAX_MODEL_LEN = 65536
MAX_OUTPUT_TOKENS = 65536 // 2
TEMPERATURE = 0.6
TOP_P = 0.95

# Paths
HF_SEED_DATASET = "G-reen/Resynth-Seed"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "init")
INITIAL_PROMPT_PATH = os.path.join(PROMPTS_DIR, "initial.txt")
EXAMPLE_FORMAT_PATH = os.path.join(PROMPTS_DIR, "example_format.txt")
EXTRA_DETAILS_DIR = os.path.join(PROMPTS_DIR, "extra_details")
EVOLVE_PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "evolve")
OUTPUT_JSONL = os.path.join(SCRIPT_DIR, "evolved_dataset.jsonl")

# Extra details count distribution (0: 40%, 1: 30%, 2: 20%, 3: 10%)
EXTRA_DETAILS_WEIGHTS = [0.2, 0.4, 0.3, 0.1]
EXTRA_DETAILS_COUNTS = [0, 1, 2, 3]

# ═══════════════════════════════════════════════════════════════════════════════
# Core Resource Loading & Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_seed_dataset():
    print(f"📦 Loading seed dataset from {HF_SEED_DATASET} ...")
    ds = load_dataset(HF_SEED_DATASET, split="train")
    seeds = [row["seed"] for row in ds]
    print(f"   Loaded {len(seeds):,} seeds.")
    return seeds

def load_extra_detail_templates() -> list[tuple[str, str]]:
    templates = []
    for fname in sorted(os.listdir(EXTRA_DETAILS_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(EXTRA_DETAILS_DIR, fname)
        templates.append((os.path.splitext(fname)[0], load_text(path)))
    return templates

def load_evolve_prompts() -> list[str]:
    """Pre-load all evolve prompts into memory to prevent heavy disk I/O."""
    default_prompt = "Please evolve this prompt further.\n\nBegin your output with ### Prompt:."
    
    if not os.path.exists(EVOLVE_PROMPTS_DIR):
        print(f"⚠️ Warning: Evolve prompts directory not found at {EVOLVE_PROMPTS_DIR}.")
        return [default_prompt]
        
    files = [f for f in os.listdir(EVOLVE_PROMPTS_DIR) if f.endswith(".txt")]
    if not files:
        print("⚠️ Warning: No evolve prompts found in directory.")
        return [default_prompt]
        
    prompts = []
    for f in files:
        content = load_text(os.path.join(EVOLVE_PROMPTS_DIR, f))
        prompts.append(f"{content}\n\nBegin your output with ### Prompt:.")
    return prompts

def _resolve_multi_option(match: re.Match) -> str:
    options = [opt.strip() for opt in match.group(1).split("/")]
    chosen = random.sample(options, random.randint(1, min(3, len(options))))
    return ", ".join(chosen)

def _resolve_single_option(match: re.Match) -> str:
    options = [opt.strip() for opt in match.group(1).split("/")]
    return random.choice(options)

def format_detail(template: str) -> str:
    result = re.sub(r"\(([^)]+)\)", _resolve_multi_option, template)
    result = re.sub(r"\[([^\]]+)\]", _resolve_single_option, result)
    return result

def build_prompt(seeds, initial_template, example_format_template, extra_detail_templates) -> str:
    chosen_seeds = random.choices(seeds, k=random.randint(1, 3))
    
    example_blocks = []
    for i, seed_text in enumerate(chosen_seeds, start=1):
        block = example_format_template.replace("{{n}}", str(i))
        block = block.replace("{{example}}", seed_text)
        example_blocks.append(block)

    examples_str = "\n\n".join(example_blocks)
    num_extra = random.choices(EXTRA_DETAILS_COUNTS, weights=EXTRA_DETAILS_WEIGHTS, k=1)[0]
    
    extra_details_str = ""
    if num_extra > 0 and extra_detail_templates:
        chosen_details = random.sample(extra_detail_templates, k=num_extra)
        formatted = [format_detail(tmpl) for _, tmpl in chosen_details]
        extra_details_str = "\n" + " ".join(formatted)

    prompt = initial_template.replace("{{examples}}", examples_str)
    prompt = prompt.replace("{{extra_details}}", extra_details_str)
    return prompt

# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Generation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_final_prompt(text: str) -> str:
    """Extracts the prompt after the designated marker."""
    if "### Prompt:" in text:
        return text.split("### Prompt:")[-1].strip()
    return text.strip()

def main():
    print("=" * 72)
    print(f"🚀 Initializing vLLM pipeline: {MODEL_ID}")
    print("=" * 72)

    # Load templates and pre-load prompts
    initial_template = load_text(INITIAL_PROMPT_PATH)
    example_format_template = load_text(EXAMPLE_FORMAT_PATH)
    extra_detail_templates = load_extra_detail_templates()
    seeds = load_seed_dataset()
    evolve_prompts_list = load_evolve_prompts()

    # Initialize vLLM with max_num_seqs to protect against OOM
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=2,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=BATCH_SIZE,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS
    )

    completed_dataset = []
    pbar = tqdm(total=TARGET_TOTAL_PROMPTS, desc="Generating Dataset")

    while len(completed_dataset) < TARGET_TOTAL_PROMPTS:
        # Determine how many items we need for this batch to avoid overshooting target heavily
        current_batch_size = min(BATCH_SIZE, TARGET_TOTAL_PROMPTS - len(completed_dataset))
        
        # State tracker for the active batch
        active_conversations = []
        for _ in range(current_batch_size):
            initial_user_msg = build_prompt(seeds, initial_template, example_format_template, extra_detail_templates)
            active_conversations.append({
                "convo": [{"role": "user", "content": initial_user_msg}],
                "evolves_left": random.randint(1, 3) # 1 initial gen + 1 to 3 evolves
            })

        # Process the static chunk until all conversations in it finish their evolve loops
        while active_conversations:
            # Format all active conversations with the chat template
            formatted_prompts = [
                tokenizer.apply_chat_template(item["convo"], tokenize=False, add_generation_prompt=True)
                for item in active_conversations
            ]
            
            # Generate next turn in parallel for all active prompts
            outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
            
            next_active_conversations = []
            
            for i, output in enumerate(outputs):
                assistant_response = output.outputs[0].text.strip()
                item = active_conversations[i]
                
                # Append assistant response to the conversation history
                item["convo"].append({"role": "assistant", "content": assistant_response})
                
                if item["evolves_left"] > 0:
                    # Needs another evolve: append new user instruction and keep in queue
                    next_evolve_msg = random.choice(evolve_prompts_list)
                    item["convo"].append({"role": "user", "content": next_evolve_msg})
                    item["evolves_left"] -= 1
                    next_active_conversations.append(item)
                else:
                    # Evolution finished: extract prompt, save, and drop from queue
                    final_text = extract_final_prompt(assistant_response)
                    completed_dataset.append({
                        "final_prompt": final_text,
                        "conversation": item["convo"]
                    })
                    pbar.update(1)
            
            # Overwrite active pool with remaining conversations for this chunk
            active_conversations = next_active_conversations
            
            # Failsafe check
            if len(completed_dataset) >= TARGET_TOTAL_PROMPTS:
                break

    pbar.close()

    print("=" * 72)
    print("💾 Saving and Uploading Dataset...")
    print("=" * 72)

    # Save to JSONL locally
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in completed_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved locally to {OUTPUT_JSONL}")

    # Push to Hugging Face
    dataset = Dataset.from_list(completed_dataset)
    dataset.push_to_hub(HF_OUTPUT_REPO)
    print(f"🎉 Successfully pushed dataset to https://huggingface.co/datasets/{HF_OUTPUT_REPO}")

if __name__ == "__main__":
    main()