import os
import re
from typing import List, Set, Optional, Tuple, Dict
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_NAME = "Kbenkhaled/Qwen3.5-27B-NVFP4" 
TP_SIZE = 2
CTX_LEN = 32768 * 2
DATASET_NAME = "G-reen/Resynth-Base"
OUTPUT_DATASET_NAME = "G-reen/Resynth-Base-Scored"
PROMPT_DELUSIONAL_PATH = "prompts/2-filter/delusional.txt"
PROMPT_TAG_PATH = "prompts/2-filter/tag.txt"
N_SAMPLES = 4
BATCH_SIZE = 1000 

def load_prompt_template(path: str) -> str:
    """Loads prompt text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_response(text: str) -> str:
    """Removes <think>...</think> blocks and trims whitespace."""
    # Use re.DOTALL to match newlines inside the think block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def parse_delusional_score(text: str) -> Optional[float]:
    """Extracts the float score from the response. Returns None if invalid."""
    text = clean_response(text)
    # Regex to find numbers (floats or integers)
    # Matches optional sign, digits, optional dot, digits
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if not matches:
        return None
    
    # Only analyze the last number found.
    # If it does not fit the format (0-10), it should be discarded.
    last_match = matches[-1]
    try:
        val = float(last_match)
        if 0 <= val <= 10:
            return val
    except ValueError:
        pass
    
    return None

def parse_tags(text: str) -> Set[str]:
    """Extracts comma-separated tags from the response."""
    text = clean_response(text)
    tags = set()
    if not text:
        return tags
    
    parts = text.split(',')
    for part in parts:
        clean_part = part.strip().lower()
        clean_part = clean_part.rstrip('.') # Remove trailing periods
        if clean_part:
            tags.add(clean_part)
    return tags

def main():
    print(f"Loading dataset {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split="train") 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Loading prompt templates...")
    if not os.path.exists(PROMPT_DELUSIONAL_PATH) or not os.path.exists(PROMPT_TAG_PATH):
        print("Error: Prompt template files not found.")
        return

    delusional_template = load_prompt_template(PROMPT_DELUSIONAL_PATH)
    tag_template = load_prompt_template(PROMPT_TAG_PATH)
    
    print(f"Initializing vLLM with model={MODEL_NAME}, tp={TP_SIZE}, ctx={CTX_LEN}...")
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TP_SIZE,
            max_model_len=CTX_LEN,
            trust_remote_code=True,
            gpu_memory_utilization=0.95 
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return
    
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_num_seqs=160,
        max_tokens=512, 
        stop=["<|endoftext|>", "<|im_end|>"] 
    )
    
    new_rows = []
    total_rows = len(ds)
    print(f"Starting processing of {total_rows} rows...")
    
    # Process in batches
    for start_idx in range(0, total_rows, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        print(f"Processing batch {start_idx} to {end_idx}...")
        
        # Select batch of rows
        batch = ds.select(range(start_idx, end_idx))
        
        all_prompts = []
        # Store metadata to map response back to row index and task
        # Format: (batch_relative_index, task_type)
        request_metadata = [] 

        for i, row in enumerate(batch):
            prompt_content = row['final_prompt']
            
            # 1. Delusional Prompts
            p_del = delusional_template.replace("{{prompt}}", prompt_content)
            for _ in range(N_SAMPLES):
                all_prompts.append(p_del)
                request_metadata.append((i, "delusional"))
                
            # 2. Tag Prompts
            p_tag = tag_template.replace("{{prompt}}", prompt_content)
            for _ in range(N_SAMPLES):
                all_prompts.append(p_tag)
                request_metadata.append((i, "tag"))
        
        # Run generation for the entire batch
        outputs = llm.generate(all_prompts, sampling_params)
        
        # Container to aggregate results per row in this batch
        batch_results = {i: {"delusional_scores": [], "tag_sets": []} for i in range(len(batch))}
        
        # Distribute outputs back to rows
        for i, output in enumerate(outputs):
            row_idx, task_type = request_metadata[i]
            generated_text = output.outputs[0].text
            
            if task_type == "delusional":
                score = parse_delusional_score(generated_text)
                if score is not None:
                    batch_results[row_idx]["delusional_scores"].append(score)
            elif task_type == "tag":
                tags = parse_tags(generated_text)
                if tags: 
                    batch_results[row_idx]["tag_sets"].append(tags)

        # Finalize rows in this batch
        for i in range(len(batch)):
            original_row = batch[i]
            results = batch_results[i]
            
            delusional_scores = results["delusional_scores"]
            tag_sets_list = results["tag_sets"]
            
            # Filter condition: invalid if no valid delusional scores were generated
            if not delusional_scores:
                continue
                
            avg_delusional_score = sum(delusional_scores) / len(delusional_scores)
            
            # Merge all tag sets
            final_tags_set = set()
            for t_set in tag_sets_list:
                final_tags_set.update(t_set)
                
            final_tags_list = sorted(list(final_tags_set))
            
            # Create new row entry
            new_row = original_row.copy()
            new_row['delusional_score'] = avg_delusional_score
            new_row['tags'] = final_tags_list
            
            new_rows.append(new_row)

    print(f"Processed {len(new_rows)} valid rows out of {total_rows} original rows.")
    
    if new_rows:
        # Create new dataset
        new_ds = Dataset.from_list(new_rows)
        
        # Push to hub
        print(f"Pushing to hub: {OUTPUT_DATASET_NAME}...")
        new_ds.push_to_hub(OUTPUT_DATASET_NAME)
        print("Done!")
    else:
        print("No valid rows were generated. Check prompt templates or model outputs.")

if __name__ == "__main__":
    main()
