#!/usr/bin/env python3
"""
SparseLoRA Experiment Runner
Runs grid search for SparseLoRA and merges results with your friend's LoRA/HiRA data.
"""

import os
import subprocess
import json
import matplotlib.pyplot as plt
import pandas as pd

# Datasets to run
DATASETS = ["sst2", "imdb", "wikitext2"]

# Search Space
RANKS = [2, 4, 8, 16, 32, 64]
LR_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0] # Grid search for LR
BASE_LR = 0.0002

# Fixed settings
EPOCHS = 3
BSZ = 32
SEED = 685
OUTPUT_DIR = "outputs/sparselora_comparison"

def run_sparselora_experiment(dataset, rank, lr):
    """
    Calls your existing train_lora_sparse.py script with specific settings.
    """
    # Calculate actual LR
    actual_lr = BASE_LR * lr
    
    print(f"Running SparseLoRA | {dataset} | r={rank} | lr={actual_lr:.6f} ...", end="", flush=True)

    cmd = [
        "python", "train_lora_sparse.py",
        "--dataset", dataset,
        "--mode", "sparselora",
        "--r", str(rank),
        "--alpha", str(rank * 2), # Standard rule: alpha = 2*rank
        "--epochs", str(EPOCHS),
        "--lr", str(actual_lr),
        "--bsz", str(BSZ),
        "--seed", str(SEED),
        "--output_dir", OUTPUT_DIR,
        "--spft_config_file", "SparseLoRA/sparselora_config.yaml" 
    ]

    try:
        # Run the script and capture output to parse the result line
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to find the "RESULT:" line
        metric = 0.0
        trainable_params = 0
        
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                # Format: dataset,model,mode,r,alpha,dropout,epochs,lr,trainable,total,metric,...
                parts = line.replace("RESULT:", "").strip().split(',')
                if len(parts) >= 11:
                    trainable_params = int(parts[8])
                    metric = float(parts[10]) # Accuracy or Perplexity
                break
        
        print(f" Done. Metric: {metric:.4f}")
        return {"metric": metric, "params": trainable_params}

    except subprocess.CalledProcessError as e:
        print(f" FAILED! Error:\n{e.stderr}")
        return {"metric": 0.0, "params": 0}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_data = []

    for dataset in DATASETS:
        print(f"\n{'='*40}\nDataset: {dataset.upper()}\n{'='*40}")
        
        for rank in RANKS:
            best_metric = 0.0
            best_lr = 0.0
            best_params = 0
            
            # Grid Search for Best LR
            for lr_mult in LR_MULTIPLIERS:
                res = run_sparselora_experiment(dataset, rank, lr_mult)
                
                # Logic: For WikiText (perplexity), lower is better. For others (accuracy), higher is better.
                current_metric = res["metric"]
                is_better = False
                
                if dataset == "wikitext2":
                    # Initialize best if 0 (first run) or check if lower
                    if current_metric > 0 and (best_metric == 0.0 or current_metric < best_metric):
                        is_better = True
                else:
                    if current_metric > best_metric:
                        is_better = True
                
                if is_better:
                    best_metric = current_metric
                    best_lr = lr_mult
                    best_params = res["params"]

            # Save the best result for this Rank
            results_data.append({
                "dataset": dataset,
                "method": "sparselora",
                "rank": rank,
                "best_lr_mult": best_lr,
                "metric": best_metric,
                "trainable_params": best_params
            })
            
            print(f"  >> BEST for r={rank}: {best_metric:.4f} (LR x{best_lr})")

 
    # Save raw data
    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "sparselora_results_summary.csv"), index=False)
    
    # Plotting
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset]
        if subset.empty: continue
        
        plt.figure(figsize=(8, 5))
        
        # Plot SparseLoRA line
        plt.plot(subset["rank"], subset["metric"], 
                 marker="^", linestyle="-", color="#3498db", linewidth=2.5, markersize=8, 
                 label="SparseLoRA")
        
        # Formatting
        plt.xscale('log', base=2)
        plt.xticks(RANKS, RANKS)
        plt.xlabel("Rank (r)")
        plt.ylabel("Perplexity" if dataset == "wikitext2" else "Accuracy")
        plt.title(f"SparseLoRA Performance: {dataset.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"sparselora_plot_{dataset}.png"))
        print(f"Plot saved for {dataset}")

if __name__ == "__main__":
    main()