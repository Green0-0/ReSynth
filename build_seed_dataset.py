#!/usr/bin/env python3
import json
import random
import time
import traceback
from collections import Counter

from datasets import load_dataset, Dataset
from langdetect import detect, LangDetectException
from huggingface_hub import hf_hub_download

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_ROWS = 10000          # desired rows per dataset
NUM_BINS = 5                # default number of length bins
MAX_STREAM = 500_000        # safety cap: max rows streamed per dataset
HF_REPO = "G-reen/Resynth-Seed"
SEED = 42

random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def is_english(text: str) -> bool:
    """Detect whether text is English using langdetect.  Uses only the first
    2000 chars for speed."""
    try:
        return detect(text[:2000]) == "en"
    except LangDetectException:
        return False


def truncate_at_punctuation(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters, preferring to cut at
    the last sentence-ending punctuation mark (.  !  ?  ;) within the allowed
    window.  Falls back to hard slice if no punctuation is found in the last
    25 % of the window."""
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    # Search backwards for a sentence-ending punctuation mark
    for i in range(len(window) - 1, max(int(max_chars * 0.75) - 1, 0), -1):
        if window[i] in '.!?;':
            return window[:i + 1]

    # No suitable punctuation found — hard-cut
    return window


def get_bin_index_dynamic(length: int, bin_edges: list[tuple[int, int]]) -> int:
    """Return the index of the bin that *length* falls into, given a list of
    (lo, hi) inclusive-on-low, exclusive-on-high pairs.  Returns -1 if out of
    range of all bins."""
    for i, (lo, hi) in enumerate(bin_edges):
        if lo <= length < hi:
            return i
        # allow the upper edge of the last bin to be inclusive
        if i == len(bin_edges) - 1 and length == hi:
            return i
    return -1


def get_bin_index(length: int, min_len: int, max_len: int, num_bins: int = NUM_BINS) -> int:
    """Return bin index 0..num_bins-1 for *length*, or -1 if out of range."""
    if length < min_len or length > max_len:
        return -1
    if length == max_len:
        return num_bins - 1
    bin_width = (max_len - min_len) / num_bins
    idx = int((length - min_len) / bin_width)
    return min(idx, num_bins - 1)


def collect_into_dynamic_bins(text_iter, bin_edges: list[tuple[int, int]],
                              source_name, max_stream=MAX_STREAM):
    """
    Stream texts from *text_iter* into bins defined by *bin_edges* — a list of
    (lo, hi) tuples where lo is inclusive and hi is exclusive.

    Each bin gets TARGET_ROWS // len(bin_edges) slots.
    Stops early once every bin is full, or after max_stream items.
    Returns (items, stats).
    """
    num_bins = len(bin_edges)
    rows_per_bin = TARGET_ROWS // num_bins
    bins = {i: [] for i in range(num_bins)}
    streamed = 0
    out_of_range = 0

    for text in text_iter:
        streamed += 1

        if streamed % 500 == 0:
            counts = [len(bins[i]) for i in range(num_bins)]
            print(f"  [{source_name}] streamed {streamed:>8,}  "
                  f"bins={counts}  oor={out_of_range}")

        length = len(text)
        bid = get_bin_index_dynamic(length, bin_edges)
        if bid < 0:
            out_of_range += 1
            continue

        if len(bins[bid]) < rows_per_bin:
            bins[bid].append(text)

        if all(len(bins[i]) >= rows_per_bin for i in range(num_bins)):
            break

        if streamed >= max_stream:
            break

    bin_sizes = [len(bins[i]) for i in range(num_bins)]

    result = []
    for i in range(num_bins):
        for text in bins[i]:
            result.append({"seed": text, "source": source_name})

    stats = {
        "source":                    source_name,
        "streamed":                  streamed,
        "filtered_out_of_range":     out_of_range,
        "bin_sizes":                 bin_sizes,
        "total_collected":           len(result),
        "target":                    TARGET_ROWS,
        "hit_target":                len(result) >= TARGET_ROWS,
        "bin_ranges":                bin_edges,
    }
    return result, stats


def collect_no_bins(text_iter, source_name, max_stream=MAX_STREAM):
    """
    Stream texts from *text_iter* without binning.  Collects up to TARGET_ROWS
    items total.
    Returns (items, stats).
    """
    collected = []
    streamed = 0

    for text in text_iter:
        streamed += 1

        if streamed % 500 == 0:
            print(f"  [{source_name}] streamed {streamed:>8,}  "
                  f"collected={len(collected)}")

        collected.append({"seed": text, "source": source_name})

        if len(collected) >= TARGET_ROWS:
            break
        if streamed >= max_stream:
            break

    stats = {
        "source":                    source_name,
        "streamed":                  streamed,
        "filtered_out_of_range":     0,
        "bin_sizes":                 [len(collected)],
        "total_collected":           len(collected),
        "target":                    TARGET_ROWS,
        "hit_target":                len(collected) >= TARGET_ROWS,
        "bin_ranges":                [],
    }
    return collected, stats


def collect_into_bins(text_iter, min_len, max_len, source_name,
                      max_stream=MAX_STREAM, num_bins=NUM_BINS):
    """
    Stream texts from *text_iter* into num_bins equal-width bins over
    [min_len, max_len].

    Stops early once every bin has rows_per_bin items, or after max_stream items.
    Returns (items, stats) where items is a list of {"seed": ..., "source": ...}
    dicts and stats is a diagnostics dict.
    """
    rows_per_bin = TARGET_ROWS // num_bins
    bins = {i: [] for i in range(num_bins)}
    streamed = 0
    out_of_range = 0

    bin_width = (max_len - min_len) / num_bins
    bin_ranges = []
    for i in range(num_bins):
        lo = min_len + i * bin_width
        hi = min_len + (i + 1) * bin_width
        bin_ranges.append((int(lo), int(hi)))

    for text in text_iter:
        streamed += 1

        if streamed % 500 == 0:
            counts = [len(bins[i]) for i in range(num_bins)]
            print(f"  [{source_name}] streamed {streamed:>8,}  "
                  f"bins={counts}  oor={out_of_range}")

        length = len(text)
        bid = get_bin_index(length, min_len, max_len, num_bins)
        if bid < 0:
            out_of_range += 1
            continue

        if len(bins[bid]) < rows_per_bin:
            bins[bid].append(text)

        # early stop when all bins full
        if all(len(bins[i]) >= rows_per_bin for i in range(num_bins)):
            break

        if streamed >= max_stream:
            break

    # collect all items from all bins (no truncation)
    bin_sizes = [len(bins[i]) for i in range(num_bins)]

    result = []
    for i in range(num_bins):
        for text in bins[i]:
            result.append({"seed": text, "source": source_name})

    stats = {
        "source":                    source_name,
        "streamed":                  streamed,
        "filtered_out_of_range":     out_of_range,
        "bin_sizes":                 bin_sizes,
        "total_collected":           len(result),
        "target":                    TARGET_ROWS,
        "hit_target":                len(result) >= TARGET_ROWS,
        "bin_ranges":                bin_ranges,
    }
    return result, stats


def extract_adjacent_blocks(text: str, min_blocks: int = 2,
                            max_blocks: int = 8) -> str | None:
    """Split *text* by newlines, pick a gaussian-centred window of adjacent
    blocks, join them, and return.  Returns None if the result is empty."""
    blocks = [b for b in text.split('\n') if b.strip()]
    if len(blocks) < min_blocks:
        return None

    # How many blocks to pick (random, clamped)
    n_pick = random.randint(min_blocks, min(max_blocks, len(blocks)))

    # Gaussian centre bias: mean at middle, std = len/6 (most of the mass in
    # the middle third)
    mean = len(blocks) / 2.0
    std = max(len(blocks) / 6.0, 1.0)
    centre = int(random.gauss(mean, std))
    # clamp to valid range
    start = centre - n_pick // 2
    start = max(0, min(start, len(blocks) - n_pick))
    end = start + n_pick

    joined = '\n'.join(blocks[start:end]).strip()
    if not joined:
        return None
    return joined


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Dataset Processors
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- 1. finepdfs-edu — truncate to 4k, no binning, min 1000 ----------

def process_finepdfs_edu():
    """HuggingFaceFW/finepdfs-edu  —  uses 'text', language column for EN filter.
    Truncate to 4000 chars, no binning, exclude < 1000 chars."""
    print("\n📄  Processing finepdfs-edu ...")
    ds = load_dataset("HuggingFaceFW/finepdfs-edu", streaming=True, split="train")
    print("  ⏳  Downloading first data shard (this may take a while) ...")

    def text_iter():
        total = 0
        lang_skip = 0
        len_skip = 0
        yielded = 0
        for row in ds:
            total += 1
            if total % 2000 == 0:
                print(f"    [finepdfs-edu raw] rows={total:>8,}  "
                      f"lang_skip={lang_skip}  len_skip={len_skip}  "
                      f"yielded={yielded}")
            doc_lang = (row.get("full_doc_lid") or "").lower()
            if not doc_lang.startswith("eng"):
                lang_skip += 1
                continue
            text = row.get("text", "")
            if not text or len(text) < 1000:
                len_skip += 1
                continue
            text = truncate_at_punctuation(text, 4000)
            if len(text) < 1000:
                len_skip += 1
                continue
            if not is_english(text):
                lang_skip += 1
                continue
            yielded += 1
            yield text

    return collect_no_bins(text_iter(), "finepdfs-edu")


# ---------- 2. finewiki — truncate to 4k, no binning, min 1000 ----------

def process_finewiki():
    """HuggingFaceFW/finewiki  —  load 'en' subset directly; text column.
    Truncate to 4000 chars, no binning, exclude < 1000 chars."""
    print("\n📖  Processing finewiki ...")
    ds = load_dataset("HuggingFaceFW/finewiki", "en", streaming=True, split="train")

    def text_iter():
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 1000:
                continue
            text = truncate_at_punctuation(text, 4000)
            if len(text) < 1000:
                continue
            if not is_english(text):
                continue
            yield text

    return collect_no_bins(text_iter(), "finewiki")


# ---------- 3. MegaScience — bins [0,500], [500,1500], exclude > 1500 ----------

def process_megascience():
    """MegaScience/MegaScience  —  question column, custom bins."""
    print("\n🔬  Processing MegaScience ...")
    ds = load_dataset("MegaScience/MegaScience", streaming=True, split="train")

    bin_edges = [(0, 500), (500, 1500)]

    def text_iter():
        for row in ds:
            question = row.get("question", "")
            if not question or len(question) > 1500:
                continue
            if not is_english(question):
                continue
            yield question

    return collect_into_dynamic_bins(text_iter(), bin_edges, "MegaScience")


# ---------- 4. writing-prompts (100-250) — merged last two bins ----------

def process_writing_prompts():
    """llm-aes/writing-prompts  —  prompt column, prepend concept prefix.

    The (100-250) bin range applies to the raw prompt length.  After binning
    we yield the prefixed text so the final seed includes the prefix."""
    print("\n✍️   Processing writing-prompts ...")
    ds = load_dataset("llm-aes/writing-prompts", streaming=True, split="train")

    PREFIX = "Write a story with the following concept:\n"

    # Custom bins — last two of the original 5 equal-width bins merged
    bin_edges = [(100, 130), (130, 160), (160, 190), (190, 250)]
    num_bins = len(bin_edges)
    rows_per_bin = TARGET_ROWS // num_bins
    bins = {i: [] for i in range(num_bins)}
    streamed = 0
    out_of_range = 0

    for row in ds:
        prompt = row.get("prompt", "")
        if not prompt or len(prompt) <= 100:
            continue
        if not is_english(prompt):
            continue

        streamed += 1
        prompt_len = len(prompt)
        bid = get_bin_index_dynamic(prompt_len, bin_edges)
        if bid < 0:
            out_of_range += 1
            continue

        if len(bins[bid]) < rows_per_bin:
            bins[bid].append(PREFIX + prompt)

        if all(len(bins[i]) >= rows_per_bin for i in range(num_bins)):
            break
        if streamed >= MAX_STREAM:
            break

        if streamed % 500 == 0:
            counts = [len(bins[i]) for i in range(num_bins)]
            print(f"  [writing-prompts] streamed {streamed:>8,}  bins={counts}")

    bin_sizes = [len(bins[i]) for i in range(num_bins)]
    result = []
    for i in range(num_bins):
        for text in bins[i]:
            result.append({"seed": text, "source": "writing-prompts"})

    stats = {
        "source":                    "writing-prompts",
        "streamed":                  streamed,
        "filtered_out_of_range":     out_of_range,
        "bin_sizes":                 bin_sizes,
        "total_collected":           len(result),
        "target":                    TARGET_ROWS,
        "hit_target":               len(result) >= TARGET_ROWS,
        "bin_ranges":                bin_edges,
    }
    return result, stats


# ---------- 5. Step-3.5-Flash-SFT — bins [0,1000],[1000,2000],[2000,3500],[3500,6000] ----------

def process_step_flash_sft():
    """stepfun-ai/Step-3.5-Flash-SFT  —  raw JSON download to avoid Arrow crash.
    Extracts first user message; filters rows with tools.
    Custom bins, exclude > 6000 chars."""
    print("\n⚡  Processing Step-3.5-Flash-SFT ...")

    bin_edges = [(0, 1000), (1000, 2000), (2000, 3500), (3500, 6000)]

    # Raw JSON shards: json/general/chunk_0.json .. chunk_99.json
    TOTAL_CHUNKS = 100

    def text_iter():
        for chunk_idx in range(TOTAL_CHUNKS):
            file_path = f"json/general/chunk_{chunk_idx}.json"
            print(f"    ↳ downloading {file_path} ...")
            try:
                local = hf_hub_download(
                    repo_id="stepfun-ai/Step-3.5-Flash-SFT",
                    filename=file_path,
                    repo_type="dataset",
                )
                with open(local, "r", encoding="utf-8") as f:
                    data = json.load(f)  # top-level array

                for example in data:
                    conversations = example.get("conversations", [])
                    if not conversations:
                        continue

                    has_tools = False
                    first_user = None
                    for msg in conversations:
                        if msg.get("role") == "user":
                            if "tools" in msg:
                                has_tools = True
                                break
                            if first_user is None:
                                first_user = msg.get("content", "")
                    if has_tools or not first_user:
                        continue

                    if len(first_user) > 6000:
                        continue

                    if is_english(first_user):
                        yield first_user

            except Exception as e:
                print(f"    ⚠  Error on {file_path}: {e}")
                continue

    return collect_into_dynamic_bins(text_iter(), bin_edges, "Step-3.5-Flash-SFT")


# ---------- 6. no_robots — no binning ----------

def process_no_robots():
    """HuggingFaceH4/no_robots  —  first user message from messages list.
    No binning."""
    print("\n🤖  Processing no_robots ...")
    ds = load_dataset("HuggingFaceH4/no_robots", streaming=True, split="train")

    def text_iter():
        for row in ds:
            messages = row.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content and is_english(content):
                        yield content
                    break  # only first user message

    return collect_no_bins(text_iter(), "no_robots")


# ---------- 7. arena-human-preference-140k — bins [0,1000],[1000,2500],[2500,5000] ----------

def _extract_arena_first_user_text(full_conversation):
    """Return the text of the first user turn whose content has no image,
    or None."""
    for turn in full_conversation:
        if not isinstance(turn, dict):
            continue

        if "user" in turn:
            msg = turn["user"]
        elif turn.get("role") == "user":
            msg = turn
        else:
            continue

        content = msg.get("content", [])
        if isinstance(content, str):
            return content if content.strip() else None

        if isinstance(content, list):
            has_image = any(
                isinstance(p, dict) and p.get("image") is not None
                for p in content
            )
            if has_image:
                return None

            texts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            combined = " ".join(t for t in texts if t).strip()
            return combined if combined else None

    return None


def process_arena():
    """lmarena-ai/arena-human-preference-140k  —  first user text from
    full_conversation (no image).  Uses language column for EN filter.
    Custom bins, exclude > 5000 chars."""
    print("\n🏟️   Processing arena-human-preference-140k ...")
    ds = load_dataset("lmarena-ai/arena-human-preference-140k",
                       streaming=True, split="train")

    bin_edges = [(0, 1000), (1000, 5000)]

    def text_iter():
        for row in ds:
            lang = (row.get("language") or "").lower()
            if lang and lang not in ("en", "eng", "english"):
                continue
            fc = row.get("full_conversation", [])
            text = _extract_arena_first_user_text(fc)
            if not text or len(text) > 5000:
                continue
            # If no language column was present, fall back to langdetect
            if not lang and not is_english(text):
                continue
            yield text

    return collect_into_dynamic_bins(text_iter(), bin_edges, "arena-human-preference-140k")


# ---------- 8. Kalomaze-Opus-Instruct-25k-filtered — bins [0,750],[750,1500],[1500,4000] ----------

def process_opus_instruct():
    """nothingiisreal/Kalomaze-Opus-Instruct-25k-filtered  —  first 'human'
    message from conversations.  Custom bins, exclude > 4000 chars."""
    print("\n💬  Processing Kalomaze-Opus-Instruct-25k-filtered ...")
    ds = load_dataset("nothingiisreal/Kalomaze-Opus-Instruct-25k-filtered",
                       streaming=True, split="train")

    bin_edges = [(0, 750), (750, 4000)]

    def text_iter():
        for row in ds:
            conversations = row.get("conversations", [])
            for msg in conversations:
                if msg.get("from") == "human":
                    value = msg.get("value", "")
                    if value and len(value) <= 4000 and is_english(value):
                        yield value
                    break  # only the first human message

    return collect_into_dynamic_bins(text_iter(), bin_edges, "Kalomaze-Opus-Instruct-25k-filtered")


# ---------- 9. WritingPrompts-Filtered (2000-6000) — KEEP THE SAME ----------

def process_writing_prompts_filtered():
    """RLAIF/WritingPrompts-Filtered  —  post_title + first long comment (>2 000
    chars).  Final text binned in 2000-6000."""
    print("\n📝  Processing WritingPrompts-Filtered ...")
    ds = load_dataset("RLAIF/WritingPrompts-Filtered",
                       streaming=True, split="train")

    def text_iter():
        for row in ds:
            post_title = (row.get("post_title") or "").strip()
            comment_texts = row.get("comment_texts", [])
            if not comment_texts or not post_title:
                continue

            selected = None
            for comment in comment_texts:
                if isinstance(comment, str) and len(comment) > 2000:
                    selected = comment
                    break
            if selected is None:
                continue

            full_text = post_title + "\n\n" + selected.strip()
            if is_english(full_text[:2000]):
                yield full_text

    return collect_into_bins(text_iter(), 2000, 6000, "WritingPrompts-Filtered")


# ---------- 10. Ultra-FineWeb — no binning, truncate 4k, min 1000 ----------

def process_ultra_fineweb():
    """openbmb/Ultra-FineWeb  —  'en' split, text in 'content', filter score >= 0.7.
    No binning, truncate to 4000 chars, exclude < 1000 chars."""
    print("\n🌐  Processing Ultra-FineWeb ...")
    ds = load_dataset("openbmb/Ultra-FineWeb", streaming=True, split="en")

    def text_iter():
        total = 0
        score_skip = 0
        len_skip = 0
        yielded = 0
        for row in ds:
            total += 1
            if total % 2000 == 0:
                print(f"    [Ultra-FineWeb raw] rows={total:>8,}  "
                      f"score_skip={score_skip}  len_skip={len_skip}  "
                      f"yielded={yielded}")

            score = row.get("score", 0)
            if score < 0.7:
                score_skip += 1
                continue

            text = row.get("content", "")
            if not text or len(text) < 1000:
                len_skip += 1
                continue

            text = truncate_at_punctuation(text, 4000)
            if len(text) < 1000:
                len_skip += 1
                continue

            yielded += 1
            yield text

    return collect_no_bins(text_iter(), "Ultra-FineWeb")


# ---------- 11. zlib — no binning, truncate 4k, min 2000, block extraction ----------

def process_zlib():
    """marianna13/zlib  —  'TEXT' column, split by newline, pick gaussian-centred
    adjacent blocks, check english.  No binning, truncate to 4000, min 2000."""
    print("\n📚  Processing zlib ...")
    ds = load_dataset("marianna13/zlib", streaming=True, split="train")

    def text_iter():
        total = 0
        for row in ds:
            total += 1
            if total % 2000 == 0:
                print(f"    [zlib raw] rows={total:>8,}")

            raw = row.get("TEXT", "")
            if not raw:
                continue

            extracted = extract_adjacent_blocks(raw)
            if extracted is None:
                continue

            extracted = truncate_at_punctuation(extracted, 4000)
            if len(extracted) < 2000:
                continue

            if not is_english(extracted):
                continue

            yield extracted

    return collect_no_bins(text_iter(), "zlib")


# ---------- 12. vault_text — no binning, truncate 4k, min 2000, block extraction ----------

def process_vault_text():
    """marianna13/vault_text  —  'TEXT' column, split by newline, pick
    gaussian-centred adjacent blocks, check english.  No binning, truncate to
    4000, min 2000."""
    print("\n🏛️   Processing vault_text ...")
    ds = load_dataset("marianna13/vault_text", streaming=True, split="train")

    def text_iter():
        total = 0
        for row in ds:
            total += 1
            if total % 2000 == 0:
                print(f"    [vault_text raw] rows={total:>8,}")

            raw = row.get("TEXT", "")
            if not raw:
                continue

            extracted = extract_adjacent_blocks(raw)
            if extracted is None:
                continue

            extracted = truncate_at_punctuation(extracted, 4000)
            if len(extracted) < 2000:
                continue

            if not is_english(extracted):
                continue

            yield extracted

    return collect_no_bins(text_iter(), "vault_text")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

PROCESSORS = [
    ("finepdfs-edu",                     process_finepdfs_edu),
    ("finewiki",                         process_finewiki),
    ("MegaScience",                      process_megascience),
    ("writing-prompts",                  process_writing_prompts),
    ("Step-3.5-Flash-SFT",              process_step_flash_sft),
    ("no_robots",                        process_no_robots),
    ("arena-human-preference-140k",      process_arena),
    ("Kalomaze-Opus-Instruct-25k",       process_opus_instruct),
    ("WritingPrompts-Filtered",          process_writing_prompts_filtered),
    ("Ultra-FineWeb",                    process_ultra_fineweb),
    ("zlib",                             process_zlib),
    ("vault_text",                       process_vault_text),
]


def main():
    print("=" * 72)
    print("  Resynth Seed Dataset Builder")
    print("=" * 72)
    print(f"  Target per dataset : {TARGET_ROWS}")
    print(f"  Default bins       : {NUM_BINS}")
    print(f"  Upload destination : {HF_REPO}")
    print("=" * 72)

    all_items = []
    all_stats = []

    for name, proc in PROCESSORS:
        t0 = time.time()
        try:
            items, stats = proc()
            elapsed = time.time() - t0
            stats["elapsed_sec"] = round(elapsed, 1)
            all_items.extend(items)
            all_stats.append(stats)
            print(f"  ✅  {name}: {stats['total_collected']} rows  "
                  f"(bins={stats['bin_sizes']})  "
                  f"[{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌  {name} FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            all_stats.append({
                "source": name,
                "error": str(e),
                "total_collected": 0,
                "hit_target": False,
                "elapsed_sec": round(elapsed, 1),
            })

    # ── shuffle ──
    random.shuffle(all_items)

    # ── build HF Dataset ──
    dataset = Dataset.from_list(all_items)

    # ══════════════════════════════════════════════════════════════════════
    # Statistics Report
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  DETAILED STATISTICS")
    print("=" * 72)

    total_rows = len(all_items)
    expected_max = TARGET_ROWS * len(PROCESSORS)
    print(f"\n  Total rows in merged dataset .... {total_rows:,}")
    print(f"  Expected maximum (all hit target)  {expected_max:,}")
    print()

    shortfalls = []
    for stats in all_stats:
        src = stats.get("source", "?")
        print(f"  ─── {src} ───")
        if "error" in stats:
            print(f"    ERROR: {stats['error']}")
            shortfalls.append((src, 0, TARGET_ROWS))
            print()
            continue

        print(f"    Rows streamed           : {stats['streamed']:>10,}")
        print(f"    Out-of-range filtered   : {stats.get('filtered_out_of_range', 0):>10,}")
        print(f"    Bin sizes               : {stats['bin_sizes']}")
        print(f"    Total collected         : {stats['total_collected']:>10,}")
        print(f"    Target                  : {stats['target']:>10,}")
        hit = stats["hit_target"]
        print(f"    Hit target?             : {'✅ YES' if hit else '❌ NO'}")
        print(f"    Elapsed                 : {stats.get('elapsed_sec', '?')}s")

        if stats.get("bin_ranges"):
            print("    Bin breakdown:")
            for i, (lo, hi) in enumerate(stats["bin_ranges"]):
                count = stats["bin_sizes"][i]
                print(f"      Bin {i} [{lo:>6,}-{hi:>6,}) : {count:>6,}")

        if not hit:
            shortfalls.append((src, stats["total_collected"], TARGET_ROWS))
        print()

    # ── Source distribution in final dataset ──
    print("  Source distribution in final dataset:")
    source_counts = Counter(item["source"] for item in all_items)
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_rows * 100 if total_rows else 0
        print(f"    {src:<45s} {cnt:>6,}  ({pct:5.1f}%)")
    print()

    # ── Shortfall warnings ──
    if shortfalls:
        print("  ⚠  DATASETS THAT DID NOT HIT TARGET:")
        for src, got, target in shortfalls:
            print(f"    • {src}: got {got:,} / {target:,}")
        print()

    # ── Length statistics ──
    lengths = [len(item["seed"]) for item in all_items]
    if lengths:
        print("  Overall length statistics (chars):")
        lengths_sorted = sorted(lengths)
        print(f"    Min    : {lengths_sorted[0]:,}")
        print(f"    Median : {lengths_sorted[len(lengths_sorted)//2]:,}")
        print(f"    Mean   : {sum(lengths)/len(lengths):,.0f}")
        print(f"    Max    : {lengths_sorted[-1]:,}")
    print()

    # ── Upload ──
    print(f"  Uploading dataset ({total_rows:,} rows) to {HF_REPO} ...")
    dataset.push_to_hub(HF_REPO)
    print("  ✅  Upload complete!")
    print("=" * 72)

    return dataset, all_stats


if __name__ == "__main__":
    main()
