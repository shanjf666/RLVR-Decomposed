import json
import argparse
import os
import numpy as np
from collections import defaultdict

def analyze_jsonl(file_path):
    if not os.path.exists(file_path):
        return None

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except:
                pass

    if not data_list:
        return None

    lengths = [len(d.get("vanilla_response", "")) for d in data_list]
    correct_lengths = [len(d.get("vanilla_response", "")) for d in data_list if d.get("label", False)]
    boxed_present = sum(1 for d in data_list if "\\boxed{" in d.get("vanilla_response", ""))
    
    return {
        "count": len(data_list),
        "lengths": lengths,
        "correct_lengths": correct_lengths,
        "boxed_count": boxed_present
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True, help="Path to model results directory")
    args = parser.parse_args()

    files_to_analyze = []
    for root, dirs, files in os.walk(args.dir_path):
        if "merged_results.jsonl" in files:
            files_to_analyze.append(os.path.join(root, "merged_results.jsonl"))

    if not files_to_analyze:
        print(f"No results found in {args.dir_path}")
        return

    print(f"{'Dataset':<60} | {'AvgLen':<8} | {'CorrLen':<8} | {'Boxed%':<6}")
    print("-" * 90)

    grand_lengths = []
    grand_correct_lengths = []
    grand_boxed = 0
    grand_count = 0

    for file in sorted(files_to_analyze):
        res = analyze_jsonl(file)
        if res:
            avg_len = np.mean(res["lengths"])
            avg_corr = np.mean(res["correct_lengths"]) if res["correct_lengths"] else 0
            boxed_rate = (res["boxed_count"] / res["count"]) * 100
            
            display_name = os.path.basename(os.path.dirname(file))
            print(f"{display_name[:60]:<60} | {avg_len:<8.1f} | {avg_corr:<8.1f} | {boxed_rate:<6.1f}%")
            
            grand_lengths.extend(res["lengths"])
            grand_correct_lengths.extend(res["correct_lengths"])
            grand_boxed += res["boxed_count"]
            grand_count += res["count"]

    if grand_count > 0:
        print("-" * 90)
        avg_len = np.mean(grand_lengths)
        avg_corr = np.mean(grand_correct_lengths) if grand_correct_lengths else 0
        boxed_rate = (grand_boxed / grand_count) * 100
        print(f"{'GRAND TOTAL':<60} | {avg_len:<8.1f} | {avg_corr:<8.1f} | {boxed_rate:<6.1f}%")

if __name__ == "__main__":
    main()
