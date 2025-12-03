import os, time, json, argparse, math, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer,
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling
)
import evaluate
from peft import LoraConfig, get_peft_model
from torch.nn.utils import prune
import torch.nn as nn

# --- SparseLoRA Import Check ---
try:
    from spft.api import SPFTConfig, get_spft_model
except ImportError:
    print("WARNING: `spft` library not found. --mode sparselora will fail.")
    print("Please install with: pip install spft")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["sst2","imdb","wikitext2"])
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default="outputs")
    
    # LoRA / SparseLoRA Hyperparameters
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    
    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--seed", type=int, default=685)
    p.add_argument("--block_size", type=int, default=256)
    
    # Mode Selection
    p.add_argument("--mode", type=str, default="sparselora", choices=["lora", "l1", "prune", "sparselora"])
    
    # SparseLoRA Specifics
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--prune_amount", type=float, default=0.5)
    p.add_argument("--spft_config_file", type=str, default=None, 
                     help="Path to the .yaml config file for SparseLoRA (spft).")
    
    return p.parse_args()

def target_modules_for_distilbert():
    return ["q_lin","k_lin","v_lin","out_lin"]

def target_modules_for_gpt2():
    return ["c_attn","c_fc","c_proj"]

def load_cls_dataset(name, tok):
    if name == "sst2":
        raw = load_dataset("glue", "sst2")
        def _tok(ex): return tok(ex["sentence"], truncation=True)
        cols_remove = ["sentence"]
        label_col = "label"
    elif name == "imdb":
        raw = load_dataset("imdb")
        def _tok(ex): return tok(ex["text"], truncation=True)
        cols_remove = ["text"]
        label_col = "label"
    else:
        raise ValueError(name)
    tokenized = raw.map(_tok, batched=True, remove_columns=cols_remove)
    return tokenized, label_col

def load_wikitext2(tok, block_size=256):
    try:
        raw = load_dataset("mindchain/wikitext2")
        print("Loaded dataset: mindchain/wikitext2")
    except Exception:
        print("mindchain/wikitext2 not found, fallback to wikitext/wikitext-2-raw-v1")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        
    def tokenize(examples):
        out = tok(examples["text"])
        return {"input_ids": out["input_ids"]}

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    def group_texts(examples):
        concatenated_ids = sum(examples["input_ids"], [])
        total_len = (len(concatenated_ids) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": []}
        result = {
            "input_ids": [
                concatenated_ids[i : i + block_size]
                for i in range(0, total_len, block_size)
            ]
        }
        return result

    chunked = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized["train"].column_names,
    )

    return chunked

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def collect_lora_params(model):
    return [p for n,p in model.named_parameters() if ("lora_A" in n or "lora_B" in n)]

class L1Trainer(Trainer):
    def __init__(self, *args, lambda_l1=0.0, **kw):
        super().__init__(*args, **kw)
        self.lambda_l1 = lambda_l1
        self._lora_params = collect_lora_params(self.model)
    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(**inputs)
        loss = out.loss
        if self.lambda_l1 > 0:
            l1 = sum(p.abs().sum() for p in self._lora_params)
            loss = loss + self.lambda_l1 * l1
        return (loss, out) if return_outputs else loss

def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(args.output_dir, "results_baseline.csv")
    
    if not os.path.exists(results_csv):
        with open(results_csv, "w") as f:
            f.write("dataset,model,adapter,r,alpha,dropout,epochs,lr,trainable_params,total_params,accuracy,train_time_min,peak_vram_gb,notes\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # --- Task: Classification (SST2 / IMDB) ---
    if args.dataset in ("sst2", "imdb"):
        ds, label_col = load_cls_dataset(args.dataset, tokenizer)
        collator = DataCollatorWithPadding(tokenizer)
        num_labels = len(set(ds["train"][label_col]))
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )

        # Manually freeze base model for SparseLoRA
        if args.mode == "sparselora":
            for param in model.parameters():
                param.requires_grad = False

        # 1. Standard LoRA
        if args.mode == "lora":
            lora_cfg = LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=target_modules_for_distilbert(),
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_cfg)

        # 2. SparseLoRA (SPFT)
        if args.mode == "sparselora":
            if not args.spft_config_file:
                raise ValueError("--spft_config_file is required for --mode sparselora")
            
            print(f"Applying SparseLoRA (SPFT) patches from: {args.spft_config_file}")
            
            spft_config = SPFTConfig.from_file(args.spft_config_file)

            # Override the dynamic settings from Command Line
            spft_config.r = args.r
            spft_config.alpha = args.alpha
            spft_config.dropout = args.dropout
            
            # Update the mode string dynamically (e.g., "svd_8" -> "svd_32")
            if hasattr(spft_config, 'mode') and "svd" in spft_config.mode:
                spft_config.mode = f"svd_{args.r}"

            model = get_spft_model(model, spft_config)

            # Unfreeze the new parameters
            print("Unfreezing SparseLoRA and Classifier parameters...")
            for name, param in model.named_parameters():
                if any(k in name for k in ["lora", "spft", "svd", "adapter"]):
                    param.requires_grad = True
                if "classifier" in name or "pre_classifier" in name or "score" in name:
                    param.requires_grad = True

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            # Fallback for SparseLoRA which doesn't have the print method
            trainable, total = count_trainable_params(model)
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": metric.compute(
                    predictions=preds, references=labels
                )["accuracy"]
            }
        
        # Handle new vs old transformers version argument
        try:
            TrainingArguments(evaluation_strategy="epoch")
            eval_kw = {"evaluation_strategy": "epoch"}
        except TypeError:
            eval_kw = {"eval_strategy": "epoch"}

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"{args.dataset}_{args.mode}"),
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            **eval_kw,
        )
        
        TrainerCls = Trainer if args.mode != "l1" else L1Trainer
        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"] if args.dataset == "sst2" else ds["test"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            **({"lambda_l1": args.lambda_l1} if args.mode == "l1" else {}),
        )

    # --- Task: Language Modeling (WikiText2) ---
    else:
        if args.model_name == "distilbert-base-uncased":
            args.model_name = "distilgpt2"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = load_wikitext2(tokenizer, block_size=args.block_size)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if args.mode == "lora":
            lora_cfg = LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=target_modules_for_gpt2(),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

        # 2. SparseLoRA (SPFT)
        if args.mode == "sparselora":
            if not args.spft_config_file:
                raise ValueError("--spft_config_file is required for --mode sparselora")
            print(f"Applying SparseLoRA (SPFT) patches from: {args.spft_config_file}")
            spft_config = SPFTConfig.from_file(args.spft_config_file)

            if hasattr(spft_config, 'r'):
                spft_config.r = args.r
            if hasattr(spft_config, 'alpha'):
                spft_config.alpha = args.alpha
            if hasattr(spft_config, 'dropout'):
                spft_config.dropout = args.dropout
            
            # Update mode string (e.g., "svd_8" -> "svd_32")
            if hasattr(spft_config, 'mode') and "svd" in spft_config.mode:
                spft_config.mode = f"svd_{args.r}"

            model = get_spft_model(model, spft_config)

            print("Unfreezing SparseLoRA parameters...")
            for name, param in model.named_parameters():
                # Unfreeze only the new adapter weights
                if any(k in name for k in ["lora", "spft", "svd", "adapter"]):
                    param.requires_grad = True

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            # Fallback for models that don't have the method
            trainable, total = count_trainable_params(model)
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

        try:
            TrainingArguments(evaluation_strategy="epoch")
            eval_kw = {"evaluation_strategy": "epoch"}
        except TypeError:
            eval_kw = {"eval_strategy": "epoch"}

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"{args.dataset}_{args.mode}"),
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="loss", 
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            **eval_kw,
        )

        TrainerCls = Trainer if args.mode != "l1" else L1Trainer
        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds.get("validation", ds.get("test")),
            tokenizer=tokenizer,
            data_collator=collator,
            **({"lambda_l1": args.lambda_l1} if args.mode == "l1" else {}),
        )

    # --- Training ---
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    trainer.train()
    
    train_time_min = (time.time() - start) / 60.0
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    # --- Pruning (if applicable) ---
    if args.mode == "prune":
        for mod_name, mod in model.named_modules():
            for pname, _ in list(mod.named_parameters(recurse=False)):
                if pname in ("lora_A.weight", "lora_B.weight"):
                    prune.l1_unstructured(mod, name=pname, amount=args.prune_amount)
                    prune.remove(mod, pname)

    # --- Save Adapter ---
    adapter_dir = os.path.join(training_args.output_dir, f"adapter_{args.mode}")
    model.save_pretrained(adapter_dir)
    print("Adapter saved to:", adapter_dir)

    # --- Evaluation & Logging ---
    metrics = trainer.evaluate()
    if args.dataset in ("sst2", "imdb"):
        acc_or_ppl = metrics.get("eval_accuracy", float("nan"))
    else:
        # Calculate Perplexity for WikiText
        eval_loss = metrics["eval_loss"]
        acc_or_ppl = math.exp(min(20, eval_loss))

    trainable, total = count_trainable_params(model)
    
    # Save to CSV 
    with open(os.path.join(args.output_dir, "results_baseline.csv"), "a") as f:
        f.write(
            f"{args.dataset},{args.model_name},{args.mode},{args.r},{args.alpha},{args.dropout},"
            f"{args.epochs},{args.lr},{trainable},{total},{acc_or_ppl:.4f},{train_time_min:.2f},{peak_vram_gb:.2f},baseline\n"
        )

    print(
        "\nRESULT:",
        f"{args.dataset},{args.model_name},{args.mode},{args.r},{args.alpha},{args.dropout},"
        f"{args.epochs},{args.lr},{trainable},{total},{acc_or_ppl:.4f},{train_time_min:.2f},{peak_vram_gb:.2f},baseline",
    )

if __name__ == "__main__":
    main()
