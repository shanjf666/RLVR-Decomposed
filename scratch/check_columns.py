from datasets import load_dataset
try:
    ds = load_dataset('math-ai/aime24', split='test')
    print("math-ai/aime24 columns:", ds.column_names)
except Exception as e:
    print("Error:", e)
    
try:
    ds2 = load_dataset('math-ai/aime25', split='test')
    print("math-ai/aime25 columns:", ds2.column_names)
except Exception as e:
    print("Error:", e)
    
try:
    ds3 = load_dataset('AI-MO/aimo-validation-amc', split='train')
    print("AI-MO/aimo-validation-amc columns:", ds3.column_names)
except Exception as e:
    print("Error:", e)
