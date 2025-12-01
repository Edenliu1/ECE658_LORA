#!/usr/bin/env python3
"""
Experiment: Compare LoRA vs HiRA across different ranks
With LR grid search to find optimal learning rate for each method
"""

import os
import subprocess
import json
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

DATASETS = ["sst2", "imdb", "wikitext2"]
EPOCHS = 3
BASE_LR = 0.0002
BSZ = 32
SEED = 685

LR_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0, 10]
METHODS = ["lora", "hira"]
RANKS = [2, 4, 8, 16, 32, 64]
OUTPUT_DIR = "outputs/lora_vs_hira"

STYLES = {
    "lora": {"color": "#2ecc71", "marker": "o", "linestyle": "-", "label": "LoRA"},
    "hira": {"color": "#e74c3c", "marker": "s", "linestyle": "--", "label": "HiRA"},
}

# =============================================================================
# Core functions
# =============================================================================

def run_experiment(dataset: str, mode: str, rank: int, lr: float) -> dict:
    """Run a single experiment."""
    cmd = [
        "python", "train_lora_hira.py",
        "--dataset", dataset, "--mode", mode, "--r", str(rank),
        "--alpha", str(rank * 2), "--epochs", str(EPOCHS),
        "--lr", str(lr), "--bsz", str(BSZ), "--seed", str(SEED),
        "--output_dir", OUTPUT_DIR,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    metric, params = 0.0, 0
    for line in output.split('\n'):
        if line.startswith("RESULT:"):
            parts = line.replace("RESULT:", "").strip().split(',')
            if len(parts) >= 11:
                metric = float(parts[10])
                params = int(parts[8])
            break
    
    return {"dataset": dataset, "mode": mode, "rank": rank, "lr": lr, 
            "metric": metric, "trainable_params": params}


def run_grid_search(dataset: str, mode: str, rank: int) -> dict:
    """Grid search over LRs, return best."""
    is_lm = dataset == "wikitext2"
    best = None
    
    print(f"  {mode.upper()} r={rank}: ", end="", flush=True)
    
    for mult in LR_MULTIPLIERS:
        lr = BASE_LR * mult
        res = run_experiment(dataset, mode, rank, lr)
        res["lr_multiplier"] = mult
        
        if best is None or (is_lm and 0 < res["metric"] < best["metric"]) or \
           (not is_lm and res["metric"] > best["metric"]):
            best = res
    
    print(f"best={best['metric']:.4f} @ LR×{best['lr_multiplier']}")
    return best


def plot_results(dataset: str, results: dict, output_dir: str):
    """Create plot for dataset."""
    is_lm = dataset == "wikitext2"
    metric_name = "Perplexity" if is_lm else "Accuracy (%)"
    
    plt.figure(figsize=(8, 5))
    
    for method in METHODS:
        valid = [r for r in results.get(method, []) if r.get("metric", 0) > 0]
        if not valid:
            continue
        
        ranks = [r["rank"] for r in valid]
        metrics = [r["metric"] if is_lm else r["metric"] * 100 for r in valid]
        style = STYLES[method]
        plt.plot(ranks, metrics, marker=style["marker"], linestyle=style["linestyle"],
                 linewidth=2.5, markersize=10, color=style["color"], label=style["label"])
    
    plt.xscale('log', base=2)
    plt.xticks(RANKS, [str(r) for r in RANKS])
    plt.xlabel('Rank (r)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'LoRA vs HiRA ({dataset.upper()}, {EPOCHS} epochs)', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"lora_vs_hira_{dataset}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, f"lora_vs_hira_{dataset}.pdf"))
    plt.close()


def create_combined_plot(all_results: dict, output_dir: str):
    """Create combined plot."""
    datasets = [d for d in DATASETS if d in all_results]
    if not datasets:
        return
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0][idx]
        results = all_results[dataset]
        is_lm = dataset == "wikitext2"
        
        for method in METHODS:
            valid = [r for r in results.get(method, []) if r.get("metric", 0) > 0]
            if not valid:
                continue
            
            ranks = [r["rank"] for r in valid]
            metrics = [r["metric"] if is_lm else r["metric"] * 100 for r in valid]
            style = STYLES[method]
            ax.plot(ranks, metrics, marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=2, markersize=8, color=style["color"], label=style["label"])
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel('Rank (r)', fontsize=11)
        ax.set_ylabel("Perplexity" if is_lm else "Accuracy (%)", fontsize=11)
        ax.set_title(dataset.upper(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'LoRA vs HiRA Comparison ({EPOCHS} epochs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lora_vs_hira_all.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "lora_vs_hira_all.pdf"))
    plt.close()


def print_summary(all_results: dict):
    """Print results table."""
    print("\n" + "="*60)
    print("RESULTS (Best LR per method)")
    print("="*60)
    
    for dataset in DATASETS:
        if dataset not in all_results:
            continue
        results = all_results[dataset]
        is_lm = dataset == "wikitext2"
        
        print(f"\n{dataset.upper()}:")
        print(f"{'Rank':<6} {'LoRA':<18} {'HiRA':<18}")
        print("-"*42)
        
        for i, rank in enumerate(RANKS):
            lora = results["lora"][i] if i < len(results["lora"]) else {"metric": 0}
            hira = results["hira"][i] if i < len(results["hira"]) else {"metric": 0}
            
            if is_lm:
                lora_s = f"{lora['metric']:.2f} (×{lora.get('lr_multiplier', '-')})"
                hira_s = f"{hira['metric']:.2f} (×{hira.get('lr_multiplier', '-')})"
            else:
                lora_s = f"{lora['metric']*100:.2f}% (×{lora.get('lr_multiplier', '-')})"
                hira_s = f"{hira['metric']*100:.2f}% (×{hira.get('lr_multiplier', '-')})"
            
            print(f"{rank:<6} {lora_s:<18} {hira_s:<18}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    
    print(f"LoRA vs HiRA | LR grid: {LR_MULTIPLIERS} | Ranks: {RANKS}")
    
    for dataset in DATASETS:
        print(f"\n{'='*40}\n{dataset.upper()}\n{'='*40}")
        
        results = {"lora": [], "hira": []}
        all_results[dataset] = results
        
        for rank in RANKS:
            for mode in METHODS:
                best = run_grid_search(dataset, mode, rank)
                results[mode].append(best)
            
            # Update plot after each rank
            plot_results(dataset, results, OUTPUT_DIR)
            create_combined_plot(all_results, OUTPUT_DIR)
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    print_summary(all_results)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
