"""src/train.py — FinBERT fine-tuning loop.

Key components:
  SentimentDataset   — PyTorch Dataset wrapping a text/label DataFrame
  FinBERTTrainer     — training loop with AdamW + linear warmup + best-ckpt saving
  run_training()     — one-call orchestrator used by Notebook 01

Training takes ~2h on GPU, ~6h on CPU for 3 epochs on ~5000 examples.
After training, the best checkpoint is saved to results/model/weights.pt.
"""

import logging
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import config
from src.model import FinBERTClassifier

logger = logging.getLogger(__name__)

NAVY = "#1a3a5c"
GOLD = "#c8a951"


# ── Dataset ──────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """PyTorch Dataset for (text, label) pairs.

    Args:
        df: DataFrame with 'text' (str) and 'label' (int 0/1/2) columns.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def _collate_fn(batch, tokenizer, device):
    """Tokenise and collate a list of (text, label) tuples into tensors."""
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=config.MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, torch.tensor(labels, dtype=torch.long, device=device)


# ── Trainer ──────────────────────────────────────────────────────────────────

class FinBERTTrainer:
    """Manages fine-tuning of FinBERTClassifier.

    Fine-tuning strategy:
      - AdamW optimiser with weight decay
      - Linear warmup over first WARMUP_RATIO of training steps
      - Gradient clipping at max_norm=1.0
      - Best checkpoint saved when validation macro-F1 improves

    Args:
        model:    Initialised FinBERTClassifier.
        train_df: Training split (text + label columns).
        val_df:   Validation split (text + label columns).
    """

    def __init__(self, model: FinBERTClassifier,
                 train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        _set_seeds(config.RANDOM_SEED)
        self.model  = model
        self.device = model.device

        collate = lambda b: _collate_fn(b, model.tokenizer, self.device)
        self.train_loader = DataLoader(SentimentDataset(train_df),
                                       batch_size=config.BATCH_SIZE,
                                       shuffle=True, collate_fn=collate)
        self.val_loader   = DataLoader(SentimentDataset(val_df),
                                       batch_size=config.BATCH_SIZE,
                                       shuffle=False, collate_fn=collate)

        self.optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        total_steps  = len(self.train_loader) * config.EPOCHS
        warmup_steps = max(1, int(config.WARMUP_RATIO * total_steps))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser, warmup_steps, total_steps)

        self.criterion   = nn.CrossEntropyLoss()
        self.best_val_f1 = -1.0
        logger.info("Trainer ready | train=%d batches | val=%d batches | "
                    "total_steps=%d | warmup=%d",
                    len(self.train_loader), len(self.val_loader),
                    total_steps, warmup_steps)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for enc, labels in tqdm(self.train_loader, desc="  train", leave=False):
            self.optimiser.zero_grad()
            loss = self.criterion(
                self.model(enc["input_ids"], enc["attention_mask"]), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _eval_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for enc, labels in tqdm(self.val_loader, desc="  val  ", leave=False):
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                total_loss += self.criterion(logits, labels).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / len(self.val_loader), val_f1

    def train(self) -> dict:
        """Run config.EPOCHS epochs.  Returns history dict."""
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}
        logger.info("Starting training for %d epochs ...", config.EPOCHS)
        t0 = time.time()

        for epoch in range(1, config.EPOCHS + 1):
            t1 = time.time()
            train_loss          = self._train_epoch()
            val_loss, val_f1    = self._eval_epoch()
            lr                  = self.scheduler.get_last_lr()[0]
            elapsed             = time.time() - t1

            print(f"  Epoch {epoch}/{config.EPOCHS} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_f1={val_f1:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_f1"].append(round(val_f1, 4))

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.model.save(config.MODEL_DIR)
                print(f"    -> New best val_f1={val_f1:.4f} saved to {config.MODEL_DIR}")

        print(f"\n  Training complete in {time.time()-t0:.0f}s | best_val_f1={self.best_val_f1:.4f}")
        return history


# ── Utilities ─────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: dict, save: bool = True) -> None:
    """Plot loss and F1 curves from the training history dict."""
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("FinBERT Fine-Tuning Curves", fontsize=13)

    ax1.plot(epochs, history["train_loss"], color=NAVY, marker="o", label="Train loss")
    ax1.plot(epochs, history["val_loss"],   color=GOLD, marker="s", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_f1"], color=NAVY, marker="o")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = config.PLOTS_DIR / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {path}")
    plt.show()


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame) -> FinBERTClassifier:
    """Fine-tune FinBERT on train_df and validate on val_df.

    Saves the best checkpoint to config.MODEL_DIR.
    Returns the loaded best model (ready for inference).

    Called from Notebook 01:
        model = run_training(train_df, val_df)
    """
    model   = FinBERTClassifier()
    trainer = FinBERTTrainer(model, train_df, val_df)
    history = trainer.train()
    plot_training_curves(history)

    # Reload best checkpoint
    best = FinBERTClassifier.load(config.MODEL_DIR)
    return best
