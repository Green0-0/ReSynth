from datasets import load_dataset
try:
    ds = load_dataset("G-reen/Resynth-Base-Scored", split="train", streaming=True)
    print("Dataset keys:", next(iter(ds)).keys())
    print("First row:", next(iter(ds)))
except Exception as e:
    print(e)
