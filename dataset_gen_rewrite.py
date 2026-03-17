#!/usr/bin/env python3
"""
Resynth Rewriting Script
Rewrites prompts to match a character persona using vLLM.
"""

import os
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

HF_SOURCE_DATASET = "G-reen/Resynth-Persona"
HF_TARGET_DATASET = "G-reen/Resynth-Prompt"
MODEL_ID = "Qwen/Qwen3.5-27B-FP8"  # As requested by user

# vLLM Config
TENSOR_PARALLEL_SIZE = 2  # Matches reference configuration
MAX_MODEL_LEN = 65536
BATCH_SIZE = 64
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "4-rewrite-message")
INIT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "init.txt")
OUTPUT_JSONL = "resynth_rewrite_dataset.jsonl"

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def construct_prompt(template, character, prompt_text):
    content = template.replace("{{character}}", character)
    content = content.replace("{{prompt}}", prompt_text)
    return content

# ═══════════════════════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading dataset: {HF_SOURCE_DATASET}")
    ds = load_dataset(HF_SOURCE_DATASET, split="train")
    
    print(f"Loading template from: {INIT_PROMPT_PATH}")
    template = load_text(INIT_PROMPT_PATH)
    
    print(f"Initializing vLLM with model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        presence_penalty=PRESENCE_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=32768
    )
    
    metadata = []
    
    print("Preparing prompts...")
    conversations = []
    for row in ds:
        original_prompt = row.get("original_prompt", "")
        new_personality = row.get("new_personality", "")
        
        # Skip if missing required fields
        if not original_prompt or not new_personality:
            continue

        full_text = construct_prompt(template, new_personality, original_prompt)
        
        # Use chat format as referenced in dataset_gen.py
        # The user model expects a chat input format (list of dicts)
        conversations.append([{"role": "user", "content": full_text}])
        
        metadata.append({
            "original_prompt": original_prompt,
            "new_personality": new_personality
        })
    
    print(f"Generating responses for {len(conversations)} items...")
    
    # Process in batches 
    outputs = llm.chat(conversations, sampling_params)
    
    results = []
    
    for i, output in enumerate(outputs):
        # In chat mode, output.outputs[0].text is the ASSISTANT content
        generated_text = output.outputs[0].text.strip()
        
        # The prompt says: "Begin your output with ### Prompt:."
        # We need to extract the content AFTER that marker.
        
        final_message = generated_text
        if "### Prompt:" in final_message:
            final_message = final_message.split("### Prompt:", 1)[1].strip()
        else:
           # Fallback: if it didn't output the marker, just take the text
           # but warn or clean up common artifacts
           final_message = generated_text
        
        # Clean up any trailing dashes or newlines
        final_message = final_message.replace("------------------------", "").strip()
        
        meta = metadata[i]
        
        results.append({
            "system_prompt": meta["new_personality"],
            "init_message": final_message,
            "original_message": meta["original_prompt"]
        })
        
    print(f"Generated {len(results)} rewrites.")
    
    if not results:
        print("No results generated. Exiting.")
        return

    # Create HF Dataset
    print("Creating HuggingFace dataset...")
    new_ds = Dataset.from_list(results)
    
    # Push to Hub
    print(f"Pushing to {HF_TARGET_DATASET}...")
    try:
        new_ds.push_to_hub(HF_TARGET_DATASET)
        print("Done!")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        # Save locally as backup
        output_path = os.path.join(SCRIPT_DIR, OUTPUT_JSONL)
        print(f"Saving locally to {output_path}")
        new_ds.to_json(output_path)

if __name__ == "__main__":
    main()
