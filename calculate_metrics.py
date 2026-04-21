import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def bootstrap_best_of_n(data, subset_size, num_iterations=1000, seed=42):
    np.random.seed(seed)
    max_vals = []
    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(data), size=subset_size, replace=True)
        sampled_data = [data[i] for i in sample_indices]
        max_vals.append(np.max(sampled_data))
    return float(np.mean(max_vals))

def evaluate_metrics(file_path, n_samples):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    grouped_data = defaultdict(list)
    for example in data:
        question_id = example["question_id"]
        if len(grouped_data[question_id]) >= n_samples:
            continue
        grouped_data[question_id].append(example)

    total_pass1 = 0.0
    total_pass16 = 0.0
    total_questions = 0
    
    for question_id, examples in grouped_data.items():
        assert len(examples) == n_samples
        labels = [1.0 if ex["label"] else 0.0 for ex in examples]
        
        pass1 = np.mean(labels)
        
        if n_samples >= 16:
            pass16 = bootstrap_best_of_n(labels, subset_size=16)
        else:
            pass16 = bootstrap_best_of_n(labels, subset_size=n_samples)
            
        total_pass1 += pass1
        total_pass16 += pass16
        total_questions += 1
        
    avg_pass1 = total_pass1 / total_questions if total_questions > 0 else 0
    avg_pass16 = total_pass16 / total_questions if total_questions > 0 else 0
    
    print(f"Questions Number: {total_questions}, Pass@1 (mean): {avg_pass1:.4f}, Pass@16 (best@16): {avg_pass16:.4f}")
    return avg_pass1, avg_pass16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    parser.add_argument("--n_samples", type=int, default=32, help="Total number of samples generated per question")
    parser.add_argument("--summary_file", type=str, default=None, help="Path to append summary results")
    parser.add_argument("--model_name", type=str, default="Unknown", help="Model name for summary")
    parser.add_argument("--dataset_name", type=str, default="Unknown", help="Dataset name for summary")
    parser.add_argument("--summary_json", type=str, default=None, help="Path to save summary results in JSON format")
    args = parser.parse_args()
   
    print(f"Calculating Metrics via Bootstrapping for {args.file_path} (n={args.n_samples})")
    print("-" * 80)
    
    pass1, pass16 = evaluate_metrics(args.file_path, args.n_samples)

    if pass1 == 0.0 and pass16 == 0.0:
        print(f"Both Pass@1 and Pass@16 are 0.0. Assuming evaluation failed or no correct answers for {args.dataset_name}. Skipping summary write.")
    else:
        if args.summary_file:
            import os
            os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
            
            duplicate_found = False
            search_str = f"Model: {args.model_name} | Dataset: {args.dataset_name} |"
            if os.path.exists(args.summary_file):
                try:
                    with open(args.summary_file, "r") as f:
                        if any(search_str in line for line in f):
                            duplicate_found = True
                except Exception:
                    pass
            
            if not duplicate_found:
                with open(args.summary_file, "a") as f:
                    f.write(f"Model: {args.model_name} | Dataset: {args.dataset_name} | Pass@1: {pass1:.4f} | Pass@16: {pass16:.4f}\n")
            else:
                print(f"Entry for {args.dataset_name} already exists in txt summary, skipping append.")

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
            
            found = False
            for e in summary_data:
                if e.get("model") == entry["model"] and e.get("dataset") == entry["dataset"]:
                    found = True
                    break
            
            if not found:
                summary_data.append(entry)
                with open(args.summary_json, "w") as f:
                    json.dump(summary_data, f, indent=4)
            else:
                print(f"Entry for {args.dataset_name} already exists in json summary, skipping append.")
