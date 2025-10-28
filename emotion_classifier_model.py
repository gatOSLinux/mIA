"""
=== MIA · Clasificador de Emociones (Pretrained Encoder + MLP) ===
- Mantiene compatibilidad con tu API pública.
- Permite usar tu TextEmbedder aleatorio (emb_dim) o un encoder preentrenado (BETO) con 768D.
- Expone freeze/unfreeze para controlar el fine-tuning desde el trainer.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel


# ==================== MÓDULO 1A: TextEmbedder (embedding aleatorio) ====================
class TextEmbedder(nn.Module):
    """
    Módulo de Embedding simple:
    - Usa el tokenizador de BETO para sub-palabras (por conveniencia, vocab, pad_id, etc.)
    - La representación es un embedding aleatorio + mean pooling (no contextual).
    """
    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        emb_dim: int = 300,
        max_length: int = 128,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.max_length = max_length

        # Capa de embedding
        self.embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.pad_id)
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            if self.pad_id is not None:
                self.embedding.weight[self.pad_id].zero_()

        # Regularización opcional (ayuda contra sobreajuste)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = batch["input_ids"].to(self.device)           # [B, T]
        attention_mask = batch["attention_mask"].to(self.device) # [B, T]

        embeds = self.embedding(input_ids)                       # [B, T, E]
        if self.training:
            embeds = self.emb_dropout(embeds)

        mask = attention_mask.bool()                             # [B, T]
        if self.cls_id is not None:
            mask = mask & (input_ids != self.cls_id)
        if self.sep_id is not None:
            mask = mask & (input_ids != self.sep_id)

        mask_f = mask.unsqueeze(-1).float()                      # [B, T, 1]
        summed = (embeds * mask_f).sum(dim=1)                    # [B, E]
        counts = mask_f.sum(dim=1).clamp(min=1.0)                # [B, 1]
        sentence_vecs = summed / counts                          # [B, E]
        return sentence_vecs

    def embed_sentence(self, text: str) -> torch.Tensor:
        return self.embed_batch([text])[0]


# ==================== MÓDULO 1B: BETOEmbedder (encoder preentrenado) ====================
class BETOEmbedder(nn.Module):
    """
    Usa el encoder de BETO (BERT en español) para obtener embeddings contextuales.
    Mean pooling sobre last_hidden_state.
    Salida: [B, 768]
    """
    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        max_length: int = 128,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.encoder.to(self.device)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        outputs = self.encoder(**inputs)  # last_hidden_state [B, T, 768]
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, 768]
        return pooled


# ==================== MÓDULO 2: MLP Classifier ====================
class MLPClassifier(nn.Module):
    """
    Feedforward para clasificación de emociones:
    Input → 128 → 64 → 6 (logits)
    """
    def __init__(
        self,
        input_dim: int = 300,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.relu1(x); x = self.dropout1(x)
        x = self.fc2(x); x = self.relu2(x); x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ==================== MÓDULO 3: Modelo Completo ====================
class EmotionClassifier(nn.Module):
    """
    Integra embedder (aleatorio o BETO) + MLP.
    - `pretrained_encoder=None` → usa TextEmbedder (emb_dim configurable)
    - `pretrained_encoder="beto"` → usa BETOEmbedder (salida 768D)
    """
    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        emb_dim: int = 300,
        max_length: int = 128,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 6,
        dropout: float = 0.3,
        device: Optional[torch.device] = None,
        pretrained_encoder: Optional[str] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_encoder == "beto":
            self.embedder = BETOEmbedder(model_name=model_name, max_length=max_length, device=self.device)
            embed_dim = 768
        else:
            self.embedder = TextEmbedder(model_name=model_name, emb_dim=emb_dim, max_length=max_length, device=self.device)
            embed_dim = emb_dim

        self.classifier = MLPClassifier(
            input_dim=embed_dim, hidden1=hidden1, hidden2=hidden2, num_classes=num_classes, dropout=dropout
        )

        self.label_map = {0: "tristeza", 1: "alegría", 2: "amor", 3: "ira", 4: "miedo", 5: "sorpresa"}

        self.to(self.device)

    # ---------- Forward & Utils ----------
    def forward(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.embedder.embed_batch(texts)  # [B, D]
        logits = self.classifier(embeddings)           # [B, C]
        return logits

    def predict(self, texts: List[str], return_probs: bool = False):
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            emotions = [self.label_map[p.item()] for p in predictions]
            if return_probs:
                return emotions, probs.cpu().numpy()
            return emotions

    def predict_single(self, text: str, return_probs: bool = False):
        out = self.predict([text], return_probs=return_probs)
        if return_probs:
            emotions, probs = out
            return emotions[0], probs[0]
        return out[0]

    # ---------- Fine-tuning helpers ----------
    def freeze_encoder(self):
        for p in self.embedder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.embedder.parameters():
            p.requires_grad = True