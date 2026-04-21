from datasets import load_dataset

datasets_to_check = [
    ("HuggingFaceH4/MATH-500", "test"),
    ("math-ai/aime24", "test"),
    ("AI-MO/aimo-validation-amc", "train"),
    ("math-ai/aime25", "test")
]

for ds_name, split in datasets_to_check:
    print(f"Checking {ds_name} ({split})...")
    try:
        ds = load_dataset(ds_name, split=split)
        print(f"Columns: {ds.column_names}")
        print(f"Example: {ds[0]}")
    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
    print("-" * 40)
