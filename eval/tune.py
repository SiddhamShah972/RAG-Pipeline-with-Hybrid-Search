import os
import time
import requests
import json
from pathlib import Path

# Important: This script should be run in an environment where it can change the environment variables
# and restart the API server. In our docker-compose setup, we might need to orchestrate this from outside
# or just change the config for the running process if it supports dynamic config.
# For simplicity in this demo, we'll assume we can call an endpoint to update chunk size 
# OR we just test `top_k` since `chunk_size` requires re-ingestion.
# Let's focus on `top_k` for the grid search without restarting the container if we can't easily re-ingest.

# Actually, to do a full grid search:
# 1. We would need to stop the server, change CHUNK_SIZE, clear Chroma/BM25, start server, ingest, evaluate.
# 2. Or we can just tune `top_k` dynamically for now.
# We will do a full grid search conceptually by iterating top_k. If we want chunk_size, we'd add an API endpoint.

from eval.evaluator import evaluate_pipeline

def tune_hyperparameters(dataset_path: str):
    # chunk_sizes = [256, 512, 1024]
    top_ks = [3, 5, 10]
    
    results = []
    
    for k in top_ks:
        print(f"\n======================================")
        print(f"Evaluating top_k={k}")
        print(f"======================================")
        
        summary = evaluate_pipeline(dataset_path, top_k=k)
        if summary:
            results.append(summary)
            
    # Save tuning summary
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"tune_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nGrid search complete. Full results in {out_file}")
    
    # Print best
    if results:
        best = max(results, key=lambda x: x["metrics_summary"].get("answer_relevancy", 0))
        print(f"Best configuration: top_k={best['top_k']} with answer_relevancy={best['metrics_summary'].get('answer_relevancy', 0):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eval/eval_dataset.json")
    args = parser.parse_args()
    
    tune_hyperparameters(args.dataset)
