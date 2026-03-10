"""src/model.py — FinBERT sentiment classifier.

The classifier wraps ProsusAI/finbert with a linear classification head
that maps the pooled [CLS] token to three sentiment classes:
    0 = negative,  1 = neutral,  2 = positive

IMPORTANT — the signal is INVERTED (contrarian):
    positive sentiment headlines  ->  signal = -1  (sell-the-news)
    negative sentiment headlines  ->  signal = +1  (buy-the-dip)

Why inverted?  Empirical ablation (using the full AlphaLens system)
showed IC = +0.019 under inversion vs IC = -0.019 with the naive map.
FinBERT on financial news captures *consensus* sentiment; consensus
positive news is already priced in, so the alpha lies in the contrarian
direction.  See Notebook 02 for the IC comparison.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

import config

logger = logging.getLogger(__name__)

# ── Label mappings ──────────────────────────────────────────────────────────

LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

# Contrarian signal: positive news -> short, negative news -> long
SIGNAL_MAP: dict[str, int] = {
    "negative":  1,   # buy the dip
    "neutral":   0,
    "positive": -1,   # sell the news
    "uncertain": 0,
}


# ── FinBERT classifier ──────────────────────────────────────────────────────

class FinBERTClassifier(nn.Module):
    """Fine-tunable three-class sentiment classifier built on ProsusAI/finbert.

    Architecture:
        BERT backbone  ->  pooler_output (768-dim)  ->  Linear(768, 3)

    Usage:
        model = FinBERTClassifier()          # loads pretrained backbone
        preds = model.predict(["Apple beats earnings"])
        # -> [{"label_name": "positive", "confidence": 0.92, ...}]
    """

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        super().__init__()
        from transformers import AutoTokenizer, BertModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.to(self.device)

    def tokenize(self, texts: list[str]) -> dict:
        """Tokenise a batch of texts into BERT input tensors."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns logits of shape (batch, 3)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)

    def predict(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of strings.

        Returns one dict per input with keys:
            text, label, label_name, confidence, probabilities
        Predictions below CONFIDENCE_THRESHOLD are labelled "uncertain".
        """
        self.eval()
        results: list[dict] = []

        for i in range(0, len(texts), config.BATCH_SIZE):
            batch = texts[i : i + config.BATCH_SIZE]
            enc   = self.tokenize(batch)

            with torch.no_grad():
                logits = self.forward(enc["input_ids"], enc["attention_mask"])

            probs = torch.softmax(logits, dim=-1).cpu()
            for j, text in enumerate(batch):
                prob_vec   = probs[j].tolist()
                confidence = float(max(prob_vec))
                pred_idx   = int(probs[j].argmax())

                if confidence >= config.CONFIDENCE_THRESHOLD:
                    label_name = ID2LABEL[pred_idx]
                    label      = pred_idx
                else:
                    label_name = "uncertain"
                    label      = LABEL2ID["neutral"]

                results.append({
                    "text":          text,
                    "label":         label,
                    "label_name":    label_name,
                    "confidence":    round(confidence, 4),
                    "probabilities": {
                        ID2LABEL[k]: round(float(prob_vec[k]), 4) for k in range(3)
                    },
                })

        return results

    def save(self, path: Path) -> None:
        """Save weights + tokenizer to directory (for loading later)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")
        self.bert.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info("Model saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "FinBERTClassifier":
        """Load a fine-tuned checkpoint from disk."""
        path = Path(path)
        weights_file = path / "weights.pt"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"No weights.pt found at {path}. Run Notebook 01 to train the model first."
            )
        # Use local HF checkpoint if available, else fall back to HuggingFace Hub
        has_hf = any((path / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))
        instance = cls(model_name=str(path) if has_hf else config.MODEL_NAME)
        instance.load_state_dict(
            torch.load(weights_file, map_location=instance.device, weights_only=False)
        )
        instance.eval()
        logger.info("Model loaded <- %s", path)
        return instance
