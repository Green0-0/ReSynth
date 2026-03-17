#!/usr/bin/env python3
"""
Resynth Roleplay Generation Script
Generates full roleplay conversations using a character persona and initial message.
"""

import os
import random
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

HF_SOURCE_DATASET = "G-reen/Resynth-Prompt"
HF_TARGET_DATASET = "G-reen/Resynth-Chat"
MODEL_ID = "Qwen/Qwen3.5-27B-FP8"

# vLLM Config (Matches dataset_gen_rewrite.py)
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 65536
BATCH_SIZE = 64
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0
ENABLE_THINKING = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_MSG_PATH = os.path.join(SCRIPT_DIR, "prompts", "5-chat", "fake_user_assistant_message.txt")
OUTPUT_JSONL = "resynth_roleplay_dataset.jsonl"

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Class
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationSession:
    def __init__(self, row_idx, data):
        self.idx = row_idx
        self.data = data
        self.system_prompt = data["system_prompt"]
        self.init_message = data["init_message"]
        self.target_rounds = random.randint(1, 4)
        self.current_rounds = 0
        
        # Side A: The Final Dataset Conversation (User = Character, Assistant = Helper)
        # Initiated with the character's first message
        self.history_main = [
            {"role": "user", "content": self.init_message}
        ]
        
        # Side B: The Simulation Conversation (User = Helper, Assistant = Character)
        # We need to load the fake prompt template to initialize this
        self.history_sim = [] # Will be initialized later
        self.sim_initialized = False

        self.is_done = False
        self.waiting_for = "assistant" # 'assistant' (Helper) or 'user' (Character/Sim)

    def init_sim_history(self, fake_msg_template):
        # Prepare the instruction message for the Simulation side
        instruction_content = fake_msg_template.replace("{{SYSTEM_PROMPT}}", self.system_prompt)
        
        self.history_sim = [
            # 1. Fake instruction message (User role in Sim)
            {"role": "user", "content": instruction_content},
            # 2. The character's first message (Assistant role in Sim)
            {"role": "assistant", "content": self.init_message}
        ]
        self.sim_initialized = True

# ═══════════════════════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════════════════════

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    print(f"Loading dataset: {HF_SOURCE_DATASET}")
    ds = load_dataset(HF_SOURCE_DATASET, split="train")
    
    print(f"Loading fake message template: {FAKE_MSG_PATH}")
    fake_msg_template = load_text(FAKE_MSG_PATH)
    
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
        max_tokens=8192 # Changed to 8k for safer roleplay length
    )
    
    # Initialize sessions
    sessions = []
    print("Initializing conversation sessions...")
    for i, row in enumerate(ds):
        if not row.get("system_prompt") or not row.get("init_message"):
            continue
        
        sess = ConversationSession(i, row)
        sess.init_sim_history(fake_msg_template)
        sessions.append(sess)
        
    print(f"Created {len(sessions)} sessions.")
    
    # Loop until all sessions are done
    step_count = 0
    while True:
        # Filter active sessions
        active_sessions = [s for s in sessions if not s.is_done]
        if not active_sessions:
            break
            
        print(f"Step {step_count}: Active sessions: {len(active_sessions)}")
        step_count += 1
        
        # ---------------------------------------------------------
        # Step: Assistant Turn (Helper)
        # ---------------------------------------------------------
        # Candidates: Sessions waiting for assistant
        asst_candidates = [s for s in active_sessions if s.waiting_for == "assistant"]
        
        if asst_candidates:
            print(f"Processing Assistant (Helper) turns: {len(asst_candidates)}")
            prompts = [s.history_main for s in asst_candidates]
            
            outputs = llm.chat(
                prompts,
                sampling_params,
                chat_template_kwargs={"enable_thinking": ENABLE_THINKING}
            )
            
            for sess, output in zip(asst_candidates, outputs):
                generated_text = output.outputs[0].text.strip()
                
                # Update Main History (Assistant responds)
                sess.history_main.append({"role": "assistant", "content": generated_text})
                
                # Update Sim History (User/Helper adds to history)
                sess.history_sim.append({"role": "user", "content": generated_text})
                
                # Switch turn
                sess.waiting_for = "user"
                
                # Check for completion (Count rounds here)
                sess.current_rounds += 1
                if sess.current_rounds >= sess.target_rounds:
                    sess.is_done = True
        
        # ---------------------------------------------------------
        # Step: User Turn (Character / Simulation)
        # ---------------------------------------------------------
        # Candidates: Sessions waiting for user, AND NOT DONE
        # (If a session was marked done above, it won't proceed to this step)
        user_candidates = [s for s in active_sessions if s.waiting_for == "user" and not s.is_done]
        
        if user_candidates:
            print(f"Processing User (Character) turns: {len(user_candidates)}")
            prompts = [s.history_sim for s in user_candidates]
            
            outputs = llm.chat(
                prompts,
                sampling_params,
                chat_template_kwargs={"enable_thinking": ENABLE_THINKING}
            )
            
            for sess, output in zip(user_candidates, outputs):
                generated_text = output.outputs[0].text.strip()
                
                # Update Sim History (Assistant/Character responds)
                sess.history_sim.append({"role": "assistant", "content": generated_text})
                
                # Update Main History (User/Character adds to history)
                sess.history_main.append({"role": "user", "content": generated_text})
                
                # Switch turn
                sess.waiting_for = "assistant"
    
    # Export
    print("Exporting results...")
    results = []
    for sess in sessions:
        results.append({
            "tree": sess.history_main, # The full conversation
            "init_message": sess.init_message,
            "user_system_prompt": sess.system_prompt
        })
        
    print("Creating HuggingFace dataset...")
    new_ds = Dataset.from_list(results)
    
    print(f"Pushing to {HF_TARGET_DATASET}...")
    try:
        new_ds.push_to_hub(HF_TARGET_DATASET)
        print("Done!")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        output_path = os.path.join(SCRIPT_DIR, OUTPUT_JSONL)
        print(f"Saving locally to {output_path}")
        new_ds.to_json(output_path)

if __name__ == "__main__":
    main()
