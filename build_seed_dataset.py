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

TARGET_ROWS = 2000          # desired rows per dataset
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


def get_bin_index(length: int, min_len: int, max_len: int, num_bins: int = NUM_BINS) -> int:
    """Return bin index 0..num_bins-1 for *length*, or -1 if out of range."""
    if length < min_len or length > max_len:
        return -1
    if length == max_len:
        return num_bins - 1
    bin_width = (max_len - min_len) / num_bins
    idx = int((length - min_len) / bin_width)
    return min(idx, num_bins - 1)


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


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Dataset Processors
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- 1. finepdfs-edu (1000-8000) ----------

def process_finepdfs_edu():
    """HuggingFaceFW/finepdfs-edu  —  uses 'text', language column for EN filter."""
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
            # full_doc_lid is the actual detected language (e.g. 'eng_Latn')
            # 'language' is just the CC dump source and is always 'eng_Latn'
            doc_lang = (row.get("full_doc_lid") or "").lower()
            if not doc_lang.startswith("eng"):
                lang_skip += 1
                continue
            text = row.get("text", "")
            if not text or not (1000 <= len(text) <= 8000):
                len_skip += 1
                continue
            yielded += 1
            yield text

    return collect_into_bins(text_iter(), 1000, 8000, "finepdfs-edu")


# ---------- 2. finewiki (1000-8000) ----------

def process_finewiki():
    """HuggingFaceFW/finewiki  —  load 'en' subset directly; text column."""
    print("\n📖  Processing finewiki ...")
    # finewiki has language subsets; load 'en' directly so no langdetect needed
    ds = load_dataset("HuggingFaceFW/finewiki", "en", streaming=True, split="train")

    def text_iter():
        for row in ds:
            text = row.get("text", "")
            if text:
                yield text

    return collect_into_bins(text_iter(), 1000, 8000, "finewiki")


# ---------- 3. MegaScience (0-1500) ----------

def process_megascience():
    """MegaScience/MegaScience  —  question column, filter < 1.5k chars."""
    print("\n🔬  Processing MegaScience ...")
    ds = load_dataset("MegaScience/MegaScience", streaming=True, split="train")

    def text_iter():
        for row in ds:
            question = row.get("question", "")
            # Check length BEFORE is_english (langdetect is slow)
            if question and len(question) <= 1500 and is_english(question):
                yield question

    return collect_into_bins(text_iter(), 0, 1500, "MegaScience", num_bins=3)


# ---------- 4. writing-prompts (100-250) ----------

def process_writing_prompts():
    """llm-aes/writing-prompts  —  prompt column, prepend concept prefix.
    
    The (100-250) bin range applies to the raw prompt length.  After binning
    we yield the prefixed text so the final seed includes the prefix."""
    print("\n✍️   Processing writing-prompts ...")
    ds = load_dataset("llm-aes/writing-prompts", streaming=True, split="train")

    PREFIX = "Write a story with the following concept:\n"

    # We need to bin on the *raw prompt* length (100-250) but yield the
    # prefixed text as the seed.  collect_into_bins bins on len(text), so
    # we do the binning manually here instead.
    num_bins = NUM_BINS
    rows_per_bin = TARGET_ROWS // num_bins
    bins = {i: [] for i in range(num_bins)}
    min_len, max_len = 100, 250
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
        bid = get_bin_index(prompt_len, min_len, max_len, num_bins)
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

    bin_width = (max_len - min_len) / num_bins
    bin_ranges = [(int(min_len + i * bin_width), int(min_len + (i+1) * bin_width))
                  for i in range(num_bins)]
    stats = {
        "source":                    "writing-prompts",
        "streamed":                  streamed,
        "filtered_out_of_range":     out_of_range,
        "bin_sizes":                 bin_sizes,
        "total_collected":           len(result),
        "target":                    TARGET_ROWS,
        "hit_target":               len(result) >= TARGET_ROWS,
        "bin_ranges":                bin_ranges,
    }
    return result, stats


# ---------- 5. Step-3.5-Flash-SFT (0-5000) ----------

def process_step_flash_sft():
    """stepfun-ai/Step-3.5-Flash-SFT  —  raw JSON download to avoid Arrow crash.
    Extracts first user message; filters rows with tools."""
    print("\n⚡  Processing Step-3.5-Flash-SFT ...")

    # Raw JSON shards: json/general/chunk_0.json .. chunk_99.json
    # Each file is a top-level JSON array of examples.
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

                    # Check for tools in any user message → skip entire example
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

                    if is_english(first_user):
                        yield first_user

            except Exception as e:
                print(f"    ⚠  Error on {file_path}: {e}")
                continue

    return collect_into_bins(text_iter(), 0, 5000, "Step-3.5-Flash-SFT")


# ---------- 6. no_robots (0-3000) ----------

def process_no_robots():
    """HuggingFaceH4/no_robots  —  first user message from messages list."""
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

    return collect_into_bins(text_iter(), 0, 3000, "no_robots", num_bins=3)


# ---------- 7. arena-human-preference-140k (0-4000) ----------

def _extract_arena_first_user_text(full_conversation):
    """Return the text of the first user turn whose content has no image,
    or None."""
    for turn in full_conversation:
        if not isinstance(turn, dict):
            continue

        # The turn structure wraps the message under a role key
        # e.g. {"user": {"role": "user", "content": [...]}}
        # or it might be flat: {"role": "user", "content": [...]}
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
                return None  # skip: first user turn has an image

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
    full_conversation (no image).  Uses language column for EN filter."""
    print("\n🏟️   Processing arena-human-preference-140k ...")
    ds = load_dataset("lmarena-ai/arena-human-preference-140k",
                       streaming=True, split="train")

    def text_iter():
        for row in ds:
            lang = (row.get("language") or "").lower()
            if lang and lang not in ("en", "eng", "english"):
                continue
            fc = row.get("full_conversation", [])
            text = _extract_arena_first_user_text(fc)
            if text:
                # If no language column was present, fall back to langdetect
                if not lang and not is_english(text):
                    continue
                yield text

    return collect_into_bins(text_iter(), 0, 4000, "arena-human-preference-140k", num_bins=4)


# ---------- 8. Kalomaze-Opus-Instruct-25k-filtered (0-3000) ----------

def process_opus_instruct():
    """nothingiisreal/Kalomaze-Opus-Instruct-25k-filtered  —  first 'human'
    message from conversations."""
    print("\n💬  Processing Kalomaze-Opus-Instruct-25k-filtered ...")
    ds = load_dataset("nothingiisreal/Kalomaze-Opus-Instruct-25k-filtered",
                       streaming=True, split="train")

    def text_iter():
        for row in ds:
            conversations = row.get("conversations", [])
            for msg in conversations:
                if msg.get("from") == "human":
                    value = msg.get("value", "")
                    if value and is_english(value):
                        yield value
                    break  # only the first human message

    return collect_into_bins(text_iter(), 0, 3000, "Kalomaze-Opus-Instruct-25k-filtered", num_bins=3)


# ---------- 9. WritingPrompts-Filtered (2000-6000) ----------

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

            # Find the first comment with length > 2000
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
