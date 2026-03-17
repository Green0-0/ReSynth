import streamlit as st
import json
import os
from datasets import load_dataset

st.set_page_config(layout="wide", page_title="ReSynth Viewer")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_hf_dataset(repo_id):
    try:
        return load_dataset(repo_id, split="train")
    except Exception as e:
        st.error(f"Error loading HF dataset: {e}")
        return None

@st.cache_data
def load_local_jsonl(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        return load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        st.error(f"Error loading local file: {e}")
        return None

def render_conversation(messages, key_prefix=""):
    """Renders a list of chat messages using Streamlit chat components."""
    if not messages:
        st.warning("No conversation history.")
        return

    for idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Color/Icon adjustments based on role if needed, 
        # but st.chat_message handles "user" and "assistant" natively.
        with st.chat_message(role):
            st.markdown(content)

def render_row_details(row, idx):
    """Renders details for a single dataset row."""
    st.header(f"Row {idx}")
    
    # Metadata Expander
    with st.expander("📝 Metadata & System Prompt", expanded=False):
        st.subheader("System Prompt (Persona)")
        st.text_area("System Prompt", row.get("user_system_prompt", ""), height=150, key=f"sp_{idx}", disabled=True)
        
        st.subheader("Initial Message")
        st.text_area("Init Message", row.get("init_message", ""), height=100, key=f"init_{idx}", disabled=True)
        
        st.subheader("Raw Data")
        st.json(row)
    
    st.divider()

    # Conversation View
    st.subheader("💬 Conversation")
    tree = row.get("tree", [])
    render_conversation(tree, key_prefix=f"chat_{idx}")

# -----------------------------------------------------------------------------
# Main APP UI
# -----------------------------------------------------------------------------

st.title("🤖 ReSynth Roleplay Dataset Viewer")

# Initialize Session State
if "data" not in st.session_state:
    st.session_state.data = None

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    source_type = st.radio("Data Source", ["Hugging Face Hub", "Local JSONL File"])
    
    if source_type == "Hugging Face Hub":
        repo_id = st.text_input("Repository ID", "G-reen/Resynth-Chat")
        if st.button("Load Dataset"):
            st.session_state.data = load_hf_dataset(repo_id)
            
    else: # Local JSONL
        default_path = os.path.join(os.getcwd(), "resynth_roleplay_dataset.jsonl")
        file_path = st.text_input("File Path", default_path)
        if st.button("Load File"):
            st.session_state.data = load_local_jsonl(file_path)

    data = st.session_state.data

    if data:
        st.success(f"✅ Loaded {len(data)} rows")
        
        st.divider()
        view_mode = st.radio("View Mode", ["Single Row Explorer", "Comparison View"])
    else:
        st.info("Please load a dataset to begin.")

# Main Content Area
if data:
    if view_mode == "Single Row Explorer":
        # Navigation
        col_nav1, col_nav2 = st.columns([1, 4])
        with col_nav1:
            row_idx = st.number_input("Row Index", min_value=0, max_value=len(data)-1, value=0, step=1)
        
        row = data[row_idx]
        render_row_details(row, row_idx)
    
    elif view_mode == "Comparison View":
        col1, col2 = st.columns(2)
        
        with col1:
            idx1 = st.number_input("Left Row Index", min_value=0, max_value=len(data)-1, value=0, key="idx1")
            render_row_details(data[idx1], idx1)
            
        with col2:
            idx2 = st.number_input("Right Row", min_value=0, max_value=len(data)-1, value=min(1, len(data)-1))
            render_row_details(data[idx2], idx2)

else:
    st.markdown("""
    ### Welcome!
    
    This tool allows you to visualize and explore the ReSynth roleplay datasets.
    
    **To get started:**
    1. Select a data source in the sidebar.
    2. Enter the Hugging Face repo ID or local file path.
    3. Explore the conversations!
    """)
