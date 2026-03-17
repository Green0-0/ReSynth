#!/usr/bin/env python3
"""
Resynth Persona Dataset Generation Script
Generates personality profiles for the Resynth dataset using vLLM.
"""

import os
import random
import json
import re
import pandas as pd
import nltk
from nltk.corpus import names as nltk_names
from datasets import load_dataset
from vllm import LLM, SamplingParams
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        print(f"Processing: {desc}")
        return iterable

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

HF_SOURCE_DATASET = "G-reen/Resynth-Base-Scored" 
HF_TARGET_DATASET = "G-reen/Resynth-Persona"
MODEL_ID = "mistralai/Mistral-Small-4-119B-2603-NVFP4"
DELUSIONAL_THRESHOLD = 0.6

# vLLM Config
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 65536
BATCH_SIZE = 16
MAX_OUTPUT_TOKENS = 65536 // 2
TEMPERATURE = 0.1
TOP_P = 0.95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "2-persona")
INIT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "init.txt")
TRAITS_PATH = os.path.join(PROMPTS_DIR, "traits.txt")
EVOLVE_DIR = os.path.join(PROMPTS_DIR, "evolve")

OUTPUT_JSONL = "resynth_persona_dataset.jsonl"

# ═══════════════════════════════════════════════════════════════════════════════
# Utils
# ═══════════════════════════════════════════════════════════════════════════════

def setup_names():
    print("Setting up name generation...")
    try:
        nltk.download('names', quiet=True)
        male_first_names = list(set(nltk_names.words('male.txt')))
        female_first_names = list(set(nltk_names.words('female.txt')))
    except LookupError:
        print("NLTK data missing, downloading...")
        nltk.download('names')
        male_first_names = list(set(nltk_names.words('male.txt')))
        female_first_names = list(set(nltk_names.words('female.txt')))
    
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/most-common-name/surnames.csv"
    try:
        df = pd.read_csv(url)
        # Use column 'name' assuming header exists, otherwise adjust based on file structure
        # The 538 dataset usually has header: name,rank,count,prop100k,...
        exhaustive_last_names = df['name'].dropna().str.title().tolist()
    except Exception as e:
        print(f"Warning: Could not load surnames from URL ({e}). Using falback.")
        exhaustive_last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", 
                                 "Miller", "Davis", "Rodriguez", "Martinez"]

    return male_first_names, female_first_names, exhaustive_last_names

def generate_exhaustive_name(male_first, female_first, last_names):
    if random.random() < 0.5:
        first = random.choice(male_first)
    else:
        first = random.choice(female_first)
    last = random.choice(last_names)
    return f"{first} {last}"

def load_traits():
    if not os.path.exists(TRAITS_PATH):
        print(f"Warning: Traits file not found at {TRAITS_PATH}")
        return ["average", "normal", "curious"]
    with open(TRAITS_PATH, 'r') as f:
        content = f.read().strip()
    # Split by comma or newline
    return [t.strip() for t in re.split(r'[,\n]+', content) if t.strip()]

def load_prompts():
    if not os.path.exists(INIT_PROMPT_PATH):
        raise FileNotFoundError(f"Init prompt not found at {INIT_PROMPT_PATH}")
        
    with open(INIT_PROMPT_PATH, 'r') as f:
        init_template = f.read().strip()
    
    evolve_templates = []
    if os.path.exists(EVOLVE_DIR):
        for fname in os.listdir(EVOLVE_DIR):
            if fname.endswith(".txt"):
                with open(os.path.join(EVOLVE_DIR, fname), 'r') as f:
                    evolve_templates.append(f.read().strip())
    
    if not evolve_templates:
        print("Warning: No evolve templates found.")
        evolve_templates = ["Please add more detail to this personality."]
        
    return init_template, evolve_templates

# ═══════════════════════════════════════════════════════════════════════════════
# Main Logic
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Setup Resources
    male_first, female_first, last_names = setup_names()
    all_traits = load_traits()
    init_template, evolve_templates = load_prompts()
    
    # 2. Init vLLM
    print(f"Initializing vLLM model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=BATCH_SIZE,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        top_p=TOP_P, 
        max_tokens=MAX_OUTPUT_TOKENS
    )
    tokenizer = llm.get_tokenizer()

    # 3. Load Dataset
    print(f"Loading dataset: {HF_SOURCE_DATASET} (streaming mode)")
    ds = load_dataset(HF_SOURCE_DATASET, split="train", streaming=True)
    
    # Output file setup
    print(f"Writing output to {OUTPUT_JSONL}")
    
    # Clear existing file
    with open(OUTPUT_JSONL, 'w') as f:
        pass 

    total_generated = 0
    
    print("Starting generation loop...")
    
    current_batch = []
    
    for row in tqdm(ds, desc="Processing Rows"):
        # Check standard keys based on inspection
        user_prompt = row.get("final_prompt") or row.get("prompt") or row.get("original_prompt")
        
        if not user_prompt:
            continue
            
        # Filter by delusion score if present
        d_score = row.get("delusional_score", 0)
        if d_score is not None and d_score > DELUSIONAL_THRESHOLD:
            continue

        # Prepare Item
        full_name = generate_exhaustive_name(male_first, female_first, last_names)
        num_traits = random.randint(1, 3)
        chosen_traits_list = random.sample(all_traits, k=min(num_traits, len(all_traits)))
        traits_str = ", ".join(chosen_traits_list)
        
        init_content = init_template.replace("{{name}}", full_name) \
                                    .replace("{{traits}}", traits_str) \
                                    .replace("{{question}}", user_prompt)
        
        # Decide evolve steps (0 to 2)
        evolve_steps = random.randint(0, 2)
        
        item = {
            "original_prompt": user_prompt,
            "tags": row.get("tags", []),
            "conversation": [{"role": "user", "content": init_content}],
            "evolve_target": evolve_steps,
            "evolve_count": 0,
            "completed": False,
            "last_response": None,
            "meta": {
                "name": full_name,
                "traits": chosen_traits_list
            }
        }
        
        current_batch.append(item)
        
        if len(current_batch) >= BATCH_SIZE:
            batch_results = []
            process_batch(llm, tokenizer, sampling_params, current_batch, evolve_templates, batch_results)
            
            # Write batch results immediately
            with open(OUTPUT_JSONL, 'a') as f:
                for res in batch_results:
                    f.write(json.dumps(res) + "\n")
            
            total_generated += len(batch_results)
            print(f"Generated {len(batch_results)} items (Total: {total_generated})...")
            
            current_batch = []

    # Process remaining
    if current_batch:
        batch_results = []
        process_batch(llm, tokenizer, sampling_params, current_batch, evolve_templates, batch_results)
        with open(OUTPUT_JSONL, 'a') as f:
            for res in batch_results:
                f.write(json.dumps(res) + "\n")
        total_generated += len(batch_results)

    print(f"Finished generation. Total items: {total_generated}")
    print(f"Saved to {OUTPUT_JSONL}")

    print("=" * 72)
    print("💾 Uploading Dataset to Hugging Face...")
    print("=" * 72)
    
    try:
        final_dataset = load_dataset("json", data_files=OUTPUT_JSONL, split="train")
        final_dataset.push_to_hub(HF_TARGET_DATASET)
        print(f"🎉 Successfully pushed dataset to https://huggingface.co/datasets/{HF_TARGET_DATASET}")
    except Exception as e:
        print(f"Error uploading to HugeFace: {e}")


def process_batch(llm, tokenizer, sampling_params, batch_items, evolve_templates, result_list):
    """
    Handles the multi-turn generation for a batch of items until all are completed.
    """
    active_items = batch_items
    
    while active_items:
        # Prepare prompts
        prompts = []
        for item in active_items:
            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(item["conversation"], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            
        # Generate
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        next_active_items = []
        
        for i, output in enumerate(outputs):
            item = active_items[i]
            generated_text = output.outputs[0].text.strip()
            
            # Update history
            item["conversation"].append({"role": "assistant", "content": generated_text})
            item["last_response"] = generated_text
            
            # Check if we need to evolve
            if item["evolve_count"] < item["evolve_target"]:
                evolve_prompt_text = random.choice(evolve_templates)
                suffixed_prompt = evolve_prompt_text + "\nBegin your output with ### Profile:, do not output anything after you finish writing the character personality profile."
                
                item["conversation"].append({"role": "user", "content": suffixed_prompt})
                item["evolve_count"] += 1
                next_active_items.append(item)
            else:
                # MARK COMPLETED
                # Extract final personality
                if "### Profile" in generated_text:
                    parts = generated_text.split("### Profile")
                    # Take the last part, handle if colon follows
                    personality = parts[-1].strip()
                    if personality.startswith(":"): 
                        personality = personality[1:].strip()
                else:
                    personality = generated_text

                # Output strictly requested columns
                result_list.append({
                    "original_prompt": item["original_prompt"],
                    "new_personality": personality,
                    "conversation": item["conversation"],
                    "tags": item["tags"]
                })
        
        active_items = next_active_items

if __name__ == "__main__":
    main()
