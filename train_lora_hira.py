#train_lora.py
import os, time, json, argparse, math, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer)
import evaluate
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn.utils import prune
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# HiRA Implementation: Hadamard High-Rank Adaptation
# Formula: h = (W ⊙ (1 + A^T @ B^T)) @ x
# =============================================================================

class HiRALayer(nn.Module):
    """
    HiRA layer that wraps a pretrained Linear or Conv1D layer.
    
    Original HiRA formula (from paper): output = (W ⊙ (1 + A^T @ B^T)) @ x + bias
    
    No scaling is applied - this matches the original author's implementation exactly.
    Supports both nn.Linear (BERT/DistilBERT) and Conv1D (GPT2).
    """
    def __init__(self, original_layer, r: int = 8):
        super().__init__()
        
        # Detect Conv1D (GPT2) vs Linear
        self.is_conv1d = hasattr(original_layer, 'nf')
        
        if self.is_conv1d:
            self.in_features = original_layer.weight.shape[0]
            self.out_features = original_layer.nf
        else:
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        
        # Keep original weight frozen
        self.weight = original_layer.weight
        self.weight.requires_grad = False
        self.bias = original_layer.bias
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # HiRA parameters
        self.r = r
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Original HiRA init: A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        x = x.to(self.lora_A.dtype)
        
        # Original HiRA: (1 + A^T @ B^T) - NO scaling applied
        _prod_AB = torch.mm(self.lora_A.T, self.lora_B.T) + 1.0
        
        if self.is_conv1d:
            adapted_weight = self.weight * _prod_AB
            size_out = x.size()[:-1] + (self.out_features,)
            result = torch.addmm(self.bias, x.view(-1, x.size(-1)), adapted_weight)
            result = result.view(size_out)
        else:
            adapted_weight = self.weight * _prod_AB.T
            result = F.linear(x, adapted_weight, self.bias)
        
        return result.to(previous_dtype)
    
    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}"


def apply_hira_to_model(model, target_modules: list, r: int = 8, modules_to_save: list = None, **kwargs):
    """
    Replace target Linear/Conv1D layers with HiRA layers.
    Also unfreezes classifier layers for classification tasks.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to replace with HiRA
        r: HiRA rank
        modules_to_save: Additional modules to keep trainable (e.g., ['classifier', 'pre_classifier'])
    """
    from transformers.pytorch_utils import Conv1D  # GPT2's Conv1D
    
    if modules_to_save is None:
        modules_to_save = []
    
    replaced = []
    
    for name, module in list(model.named_modules()):
        # Check if this module name matches any target
        if not any(target in name for target in target_modules):
            continue
        
        # Support both Linear and Conv1D (GPT2)
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue
        
        # Get parent module and attribute name
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        
        # Create HiRA layer (original implementation, no scaling)
        hira_layer = HiRALayer(module, r=r)
        hira_layer = hira_layer.to(module.weight.device).to(module.weight.dtype)
        
        # Replace
        setattr(parent, attr_name, hira_layer)
        replaced.append(name)
    
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze HiRA parameters (lora_A, lora_B)
    for n, p in model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad = True
    
    # Unfreeze modules_to_save (e.g., classifier layers)
    for n, p in model.named_parameters():
        if any(mod_name in n for mod_name in modules_to_save):
            p.requires_grad = True
    
    print(f"[HiRA] Replaced {len(replaced)} layers: {replaced[:4]}...")
    if modules_to_save:
        print(f"[HiRA] Also training: {modules_to_save}")
    return model


def print_trainable_parameters_hira(model):
    """Print trainable parameters for HiRA model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
    return trainable, total

# ============================================================================= 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["sst2","imdb","wikitext2"])
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default="outputs")
    # LoRA
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    # train
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--seed", type=int, default=685)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--mode", type=str, default="lora", choices=["lora","hira","l1","prune"])
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--prune_amount", type=float, default=0.5)
    return p.parse_args()

def target_modules_for_distilbert():
    # return ["q_lin","v_lin"]
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
    from datasets import load_dataset
    try:
        raw = load_dataset("wikitext", "wikitext-2-v1") # load Hugging Face's official wikitext2_v1 dataset
        print("Loaded dataset: wikitext/wikitext-2-v1")
    except Exception:
        print("mindchain/wikitext2 not found, fallback to wikitext/wikitext-2-raw-v1")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tokenize(examples):
        # return only input_ids, no attention_mask
        out = tok(examples["text"])
        return {"input_ids": out["input_ids"]}

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],  # drop original text column
    )
    def group_texts(examples):
        # examples["input_ids"] is a list of lists
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
# train for sst2 and imdb
    if args.dataset in ("sst2", "imdb"):
        ds, label_col = load_cls_dataset(args.dataset, tokenizer)
        collator = DataCollatorWithPadding(tokenizer)
        num_labels = len(set(ds["train"][label_col]))
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )

        # Apply LoRA or HiRA
        if args.mode == "hira":
            model = apply_hira_to_model(
                model,
                target_modules=target_modules_for_distilbert(),
                r=args.r,
                modules_to_save=["classifier", "pre_classifier"],  # Train classifier for cls tasks
            )
            print_trainable_parameters_hira(model)
        else:
            # LoRA for DistilBERT
            lora_cfg = LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=target_modules_for_distilbert(),
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": metric.compute(
                    predictions=preds, references=labels
                )["accuracy"]
            }

        try:
            TrainingArguments(evaluation_strategy="epoch")
            eval_kw = {"evaluation_strategy": "epoch"}
        except TypeError:
            eval_kw = {"eval_strategy": "epoch"}

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"{args.dataset}_{args.mode}"),
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            learning_rate=args.lr,  # Same LR for fair comparison
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
#train for wikitext2
    else:
        if args.model_name == "distilbert-base-uncased":
            args.model_name = "distilgpt2"
        from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = load_wikitext2(tokenizer, block_size=args.block_size)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Apply LoRA or HiRA
        if args.mode == "hira":
            model = apply_hira_to_model(
                model,
                target_modules=target_modules_for_gpt2(),
                r=args.r,
                modules_to_save=[],  # No extra modules for causal LM
            )
            print_trainable_parameters_hira(model)
        else:
            lora_cfg = LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=target_modules_for_gpt2(),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        try:
            TrainingArguments(evaluation_strategy="epoch")
            eval_kw = {"evaluation_strategy": "epoch"}
        except TypeError:
            eval_kw = {"eval_strategy": "epoch"}

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"{args.dataset}_{args.mode}"),
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            learning_rate=args.lr,  # Same LR for fair comparison
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

    if args.mode == "prune":
        for mod_name, mod in model.named_modules():
            for pname, _ in list(mod.named_parameters(recurse=False)):
                if pname in ("lora_A.weight", "lora_B.weight"):
                    prune.l1_unstructured(mod, name=pname, amount=args.prune_amount)
                    prune.remove(mod, pname)

    adapter_dir = os.path.join(training_args.output_dir, f"adapter_{args.mode}")
    
    # Save model: PEFT models use save_pretrained, HiRA saves state dict
    if args.mode == "hira":
        os.makedirs(adapter_dir, exist_ok=True)
        # Save only HiRA parameters
        hira_state = {k: v for k, v in model.state_dict().items() if 'lora_A' in k or 'lora_B' in k}
        torch.save(hira_state, os.path.join(adapter_dir, "hira_adapter.pt"))
        # Save config
        config = {"r": args.r, "alpha": args.alpha, "dropout": args.dropout, "mode": "hira"}
        with open(os.path.join(adapter_dir, "hira_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    else:
        model.save_pretrained(adapter_dir)

    metrics = trainer.evaluate()
    if args.dataset in ("sst2", "imdb"):
        acc_or_ppl = metrics.get("eval_accuracy", float("nan"))
    else:
        eval_loss = metrics["eval_loss"]
        acc_or_ppl = math.exp(min(20, eval_loss))

    trainable, total = count_trainable_params(model)
    
    # Log results with correct adapter name
    adapter_name = args.mode
    with open(os.path.join(args.output_dir, "results_baseline.csv"), "a") as f:
        f.write(
            f"{args.dataset},{args.model_name},{adapter_name},{args.r},{args.alpha},{args.dropout},"
            f"{args.epochs},{args.lr},{trainable},{total},{acc_or_ppl:.4f},{train_time_min:.2f},{peak_vram_gb:.2f},baseline\n"
        )

    print(
        "\nRESULT:",
        f"{args.dataset},{args.model_name},{adapter_name},{args.r},{args.alpha},{args.dropout},"
        f"{args.epochs},{args.lr},{trainable},{total},{acc_or_ppl:.4f},{train_time_min:.2f},{peak_vram_gb:.2f},baseline",
    )
    print("Adapter saved to:", adapter_dir)
if __name__ == "__main__":
    main()