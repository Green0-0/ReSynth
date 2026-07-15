# ReSynth

ReSynth is a pipeline for synthesizing large, varied conversational datasets for LLM
training. It starts from real-world text and human-written prompts, evolves and filters
them with an LLM judge, layers on generated personas, and finally produces full
multi-turn conversations ready for fine-tuning.

## Pipeline

The scripts are meant to be run in order, with each stage reading the previous stage's
output from the Hugging Face Hub and pushing its own result back up:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `build_seed_dataset.py` | Public corpora (FineWiki, FinePDFs-Edu, MegaScience, WritingPrompts, No Robots, LMArena human preference, Kalomaze-Opus-Instruct, Ultra-FineWeb, zlib, the Vault, etc.) | `G-reen/Resynth-Seed` |
| 2 | `dataset_gen.py` | `G-reen/Resynth-Seed` | `G-reen/Resynth-Base` |
| 3 | `score_dataset.py` | `G-reen/Resynth-Base` | `G-reen/Resynth-Base-Scored` |
| 4 | `dataset_gen_persona.py` | `G-reen/Resynth-Base-Scored` | `G-reen/Resynth-Persona` |
| 5 | `dataset_gen_rewrite.py` | `G-reen/Resynth-Persona` | `G-reen/Resynth-Prompt` |
| 6 | `dataset_gen_roleplay.py` | `G-reen/Resynth-Prompt` | `G-reen/Resynth-Chat` |

- **`build_seed_dataset.py`** streams a mix of public datasets, filters to English text,
  buckets documents by length, and assembles a balanced seed set uploaded to the Hub.
- **`dataset_gen.py`** turns seed documents into initial conversation prompts and evolves
  each one 1-3 times (adding detail, difficulty, agentic framing, etc.) using a local
  vLLM-served model, guided by the templates in `prompts/0-init` and `prompts/1-evolve`.
- **`score_dataset.py`** runs an LLM-as-judge pass over generated prompts, scoring them
  (e.g. a "delusional" score) and tagging them so low-quality or off-distribution rows
  can be filtered out, using `prompts/2-filter`.
- **`dataset_gen_persona.py`** invents character personas (names, traits, backstory) for
  the surviving prompts, driven by `prompts/3-persona`.
- **`dataset_gen_rewrite.py`** rewrites each prompt in the voice of its assigned persona
  using `prompts/4-rewrite-message/init.txt`.
- **`dataset_gen_roleplay.py`** simulates a full back-and-forth conversation between the
  persona and an assistant, producing multi-turn chats via `prompts/5-chat`.
- **`visualize_dataset.py`** is a Streamlit app for browsing any of the intermediate or
  final datasets — system prompt, persona, and full conversation — either from the Hub
  or a local JSONL file.

`agent_instructions/` holds the working notes and instructions used to direct the coding
agents that built out each stage of this pipeline.

## Requirements

The generation scripts (`dataset_gen*.py`, `score_dataset.py`) expect an environment with
`vllm`, `datasets`, `huggingface_hub`, `pandas`, and `nltk` installed, plus access to a GPU
large enough for the configured model and tensor-parallel size. `build_seed_dataset.py`
additionally needs `langdetect`. `visualize_dataset.py` requires `streamlit`.

```bash
pip install vllm datasets huggingface_hub pandas nltk langdetect streamlit
```

Each script has its model ID, Hugging Face repo names, and vLLM configuration
(tensor-parallel size, context length, batch size, etc.) set as constants near the top of
the file — adjust these for your own hardware and target Hub repos before running.

## Usage

Each stage is a standalone script:

```bash
python build_seed_dataset.py
python dataset_gen.py
python score_dataset.py
python dataset_gen_persona.py
python dataset_gen_rewrite.py
python dataset_gen_roleplay.py
```

To inspect a dataset at any stage:

```bash
streamlit run visualize_dataset.py
```

## Repository layout

```
agent_instructions/   Notes and prompts used to direct the agents that built this pipeline
prompts/               Prompt templates consumed by the generation scripts
  0-init/               Initial prompt construction + extra-detail modifiers
  1-evolve/             Prompt evolution strategies (harder, agentic, roleplay, etc.)
  2-filter/             LLM-judge scoring and tagging prompts
  3-persona/            Persona generation and evolution prompts
  4-rewrite-message/    Persona-conditioned prompt rewriting
  5-chat/               Templates for simulating full conversations
build_seed_dataset.py
dataset_gen.py
dataset_gen_persona.py
dataset_gen_rewrite.py
dataset_gen_roleplay.py
score_dataset.py
visualize_dataset.py
```

## Notes

This is a research pipeline under active development — model IDs, Hub repo names, and
prompt templates are being iterated on, so expect the constants at the top of each script
to change between runs.
