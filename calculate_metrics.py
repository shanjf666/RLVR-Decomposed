import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def unbiased_pass_at_k_accuracy(file_path, k, n):
    # Read the JSONLines file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Group predictions by question_id
    grouped_data = defaultdict(list)
    for example in data:
        question_id = example["question_id"]
        if len(grouped_data[question_id]) >= n:
            continue
        grouped_data[question_id].append(example)

    # Calculate unbiased pass@K accuracy
    total_pass_k_prob = 0
    total_questions = 0
    
    for question_id, examples in grouped_data.items():
        # Count correct and total solutions for this question
        assert len(examples) == n  # Total number of samples
        c = sum(1 for ex in examples if ex["label"])  # Number of correct samples
        # Apply unbiased pass@k formula
        assert n >= k
        # Calculate unbiased pass@k probability
        if n - c < k:
            prob = 1.0
        else:
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
            
        total_pass_k_prob += prob
        total_questions += 1
    
    # Calculate average pass@K probability across all questions
    avg_pass_k = total_pass_k_prob / total_questions if total_questions > 0 else 0
    print(f"Questions Number: {total_questions}, Unbiased Pass@{k}/{n}: {avg_pass_k}")
    return avg_pass_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    parser.add_argument("--n_samples", type=int, default=256, help="Total number of samples generated per question")
    parser.add_argument("--summary_file", type=str, default=None, help="Path to append summary results")
    parser.add_argument("--model_name", type=str, default="Unknown", help="Model name for summary")
    parser.add_argument("--dataset_name", type=str, default="Unknown", help="Dataset name for summary")
    parser.add_argument("--summary_json", type=str, default=None, help="Path to save summary results in JSON format")
    args = parser.parse_args()
   
    Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Filter Ks to only those <= n_samples
    Ks = [k for k in Ks if k <= args.n_samples]
    
    results = {}
    print(f"Calculating Unbiased Pass@K for {args.file_path} (n={args.n_samples})")
    for K in tqdm(Ks):
        print("-" * 80)
        test_file = args.file_path
        unbiased_pass_k_accuracy = unbiased_pass_at_k_accuracy(test_file, k=K, n=args.n_samples)
        results[K] = unbiased_pass_k_accuracy

    pass1 = results.get(1, 0.0)
    pass16 = results.get(16, 0.0)

    if args.summary_file:
        import os
        os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
        with open(args.summary_file, "a") as f:
            f.write(f"Model: {args.model_name} | Dataset: {args.dataset_name} | Pass@1: {pass1:.4f} | Pass@16: {pass16:.4f}\n")

    if args.summary_json:
        import os
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        summary_data = []
        if os.path.exists(args.summary_json):
            try:
                with open(args.summary_json, "r") as f:
                    summary_data = json.load(f)
            except Exception:
                pass
        
        entry = {
            "model": args.model_name,
            "dataset": args.dataset_name,
            "pass@1": pass1,
            "pass@16": pass16
        }
        
        # Avoid duplicate entries for same model and dataset
        found = False
        for e in summary_data:
            if e.get("model") == entry["model"] and e.get("dataset") == entry["dataset"]:
                e.update(entry)
                found = True
                break
        
        if not found:
            summary_data.append(entry)
            
        with open(args.summary_json, "w") as f:
            json.dump(summary_data, f, indent=4)
