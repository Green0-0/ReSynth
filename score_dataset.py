import os
import re
from typing import List, Set, Optional, Tuple, Dict
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3.5-27B-FP8" 
TP_SIZE = 4
CTX_LEN = 32768 * 2
DATASET_NAME = "G-reen/Resynth-Base"
OUTPUT_DATASET_NAME = "G-reen/Resynth-Base-Scored"
PROMPT_DELUSIONAL_PATH = "prompts/2-filter/delusional.txt"
PROMPT_TAG_PATH = "prompts/2-filter/tag.txt"
N_SAMPLES = 8
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
    
    # Check for header
    header = "### Delusional Score:"
    if header in text:
        # Split by header and take the part after it
        parts = text.split(header)
        # Take the last part in case header appears multiple times (unlikely but safe)
        content_after_header = parts[-1].strip()
        
        # Now find numbers in this content
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", content_after_header)
        if not matches:
            return None
            
        # If there are multiple numbers after the header, ignore the response
        if len(matches) > 1:
            return None
            
        # Process the single number
        try:
            val = float(matches[0])
            if 0 <= val <= 10:
                return val
        except ValueError:
            pass
            
        return None
    
    return None

def parse_tags(text: str) -> Set[str]:
    """Extracts comma-separated tags from the response."""
    text = clean_response(text)
    tags = set()
    
    header = "### Tags:"
    content_to_parse = "" # Default to empty if header not found
    
    if header in text:
        parts = text.split(header)
        content_to_parse = parts[-1].strip()
    
    # If header mandated, we strictly return empty if not found.
    # The user said "only accept the tags after ### Tags:"
    # This implies we should ignore text before it or if it's missing.
    
    if not content_to_parse:
        return tags
    
    parts = content_to_parse.split(',')
    for part in parts:
        clean_part = part.strip().lower()
        clean_part = clean_part.rstrip('.') # Remove trailing periods
        
        # Remove leading headers in form of *header*:
        # This regex matches optional whitespace, followed by *text*:, followed by optional whitespace
        clean_part = re.sub(r'^\s*\*.*?\*:\s*', '', clean_part)
        
        # Check word count
        word_count = len(clean_part.split())
        
        if clean_part and word_count <= 5:
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
            gpu_memory_utilization=0.85,
            max_num_seqs=128,
            max_num_batched_tokens=8192
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    # Get tokenizer from vLLM
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=4096*2, 
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
            p_del_content = delusional_template.replace("{{prompt}}", prompt_content)
            # Apply chat template
            p_del_messages = [{"role": "user", "content": p_del_content}]
            p_del = tokenizer.apply_chat_template(p_del_messages, tokenize=False, add_generation_prompt=True)

            for _ in range(N_SAMPLES):
                all_prompts.append(p_del)
                request_metadata.append((i, "delusional"))
                
            # 2. Tag Prompts
            p_tag_content = tag_template.replace("{{prompt}}", prompt_content)
            # Apply chat template
            p_tag_messages = [{"role": "user", "content": p_tag_content}]
            p_tag = tokenizer.apply_chat_template(p_tag_messages, tokenize=False, add_generation_prompt=True)

            for _ in range(N_SAMPLES):
                all_prompts.append(p_tag)
                request_metadata.append((i, "tag"))
        
        # Run generation for the entire batch
        outputs = llm.generate(all_prompts, sampling_params)
        
        # Container to aggregate results per row in this batch
        batch_results = {i: {"delusional_scores": [], "tag_sets": [], "raw_delusional": [], "raw_tags": []} for i in range(len(batch))}
        
        # Distribute outputs back to rows
        for i, output in enumerate(outputs):
            row_idx, task_type = request_metadata[i]
            generated_text = output.outputs[0].text
            
            if task_type == "delusional":
                batch_results[row_idx]["raw_delusional"].append(generated_text)
                score = parse_delusional_score(generated_text)
                if score is not None:
                    batch_results[row_idx]["delusional_scores"].append(score)
            elif task_type == "tag":
                batch_results[row_idx]["raw_tags"].append(generated_text)
                tags = parse_tags(generated_text)
                if tags: 
                    batch_results[row_idx]["tag_sets"].append(tags)

        # Finalize rows in this batch
        for i in range(len(batch)):
            original_row = batch[i]
            results = batch_results[i]
            
            delusional_scores = results["delusional_scores"]
            tag_sets_list = results["tag_sets"]
            raw_delusional_responses = results["raw_delusional"]
            raw_tag_responses = results["raw_tags"]
            
            # Filter condition: invalid if no valid delusional scores were generated
            # We still might want to save the raw responses even if scoring failed for debugging?
            # User said "Filter out responses that are improperly generated" in first turn. 
            # In update, user just asks to "record the responses". 
            # If we skip the row, we lose the recorded responses. 
            # So maybe we keep the row but mark score as invalid? 
            # Or we stick to the filter logic.
            # "filter out responses that are improperly generated" implies dropping the row.
            
            # The user now asks: "If no scores delusional scores were generated or no tags were found you should use a NaN or None or whatever the default null value for hf is"
            
            if not delusional_scores:
                avg_delusional_score = None # Hugging Face datasets handles None as null
                max_delusional_score = None
                all_delusional_scores = None
            else:
                avg_delusional_score = sum(delusional_scores) / len(delusional_scores)
                max_delusional_score = max(delusional_scores)
                all_delusional_scores = delusional_scores
            
            # Merge all tag sets
            final_tags_set = set()
            for t_set in tag_sets_list:
                final_tags_set.update(t_set)
            
            if final_tags_set:
                final_tags_list = sorted(list(final_tags_set))
            else:
                final_tags_list = None
            
            # Create new row entry
            new_row = original_row.copy()
            new_row['delusional_score'] = avg_delusional_score
            new_row['max_delusional_score'] = max_delusional_score
            new_row['all_delusional_scores'] = all_delusional_scores
            new_row['tags'] = final_tags_list
            new_row['raw_delusional_responses'] = raw_delusional_responses
            new_row['raw_tag_responses'] = raw_tag_responses
            
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
