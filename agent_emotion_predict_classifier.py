"""
=== MIA · Agent Emotion Predict Classifier (Text/BETO Embedder + MLP) ===
Objetivo: predecir la emoción del AGENTE (label_agent: 0..5) a partir de:
  - el TEXTO del usuario
  - la EMOCIÓN del texto (label del usuario: 0..5)

Arquitectura:
  Texto ──▶ Embedder (TextEmbedder ó BETOEmbedder) ─▶ h_text ∈ R^D
  Label usuario (0..5) ─▶ one-hot(6) ─▶ (feature dropout opcional)
  Concatenación [h_text ; onehot_label] ─▶ MLP ─▶ logits (6)

Notas:
- Si usas BETOEmbedder, se recomienda congelarlo (freeze) para esta segunda red.
- El feature dropout en la one-hot del label obliga al modelo a mirar el TEXTO en los casos ambiguos.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from emotion_classifier_model import TextEmbedder, BETOEmbedder  # reemplaza con tu import real


class FeatureDropout(nn.Module):
    """Apaga aleatoriamente (con prob p) TODA la rama de la one-hot del label en entrenamiento.
    Si p=0.2, en el 20% de los batches el modelo debe decidir solo con el texto.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # Con prob p, zerea todo el vector (por muestra)
        mask = (torch.rand(x.size(0), 1, device=x.device) > self.p).float()
        return x * mask


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 64, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.out(x)


class AgentEmotionPredictClassifier(nn.Module):
    """
    Segunda red: predice la emoción del AGENTE (0..5) a partir de (texto, label_usuario).

    Parámetros clave:
      - pretrained_encoder: None → TextEmbedder (emb_dim)
                            "beto" → BETOEmbedder (768D)
      - label_feature_dropout: apaga la one-hot a veces para forzar al modelo a usar el texto en casos ambiguos.
    """
    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        pretrained_encoder: Optional[str] = "beto",
        emb_dim: int = 300,
        max_length: int = 128,
        hidden1: int = 256,
        hidden2: int = 64,
        num_classes: int = 6,
        dropout: float = 0.2,
        label_feature_dropout: float = 0.15,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_encoder == "beto":
            self.embedder = BETOEmbedder(model_name=model_name, max_length=max_length, device=self.device)
            embed_dim = 768
        else:
            self.embedder = TextEmbedder(model_name=model_name, emb_dim=emb_dim, max_length=max_length, device=self.device)
            embed_dim = emb_dim

        self.label_dim = 6  # one-hot(6)
        self.feat_drop = FeatureDropout(p=label_feature_dropout)
        self.classifier = MLP(input_dim=embed_dim + self.label_dim,
                      hidden1=hidden1, hidden2=hidden2,
                      num_classes=num_classes, dropout=dropout)  # num_classes = salida del AGENTE (ahora 2)
        self.to(self.device)

    # ---------- Utils ----------
    @staticmethod
    def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        # labels: [B] int64 → one-hot [B, C]
        return F.one_hot(labels.long(), num_classes=num_classes).float()

    def freeze_encoder(self):
        for p in self.embedder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.embedder.parameters():
            p.requires_grad = True

    # ---------- Forward / Predict ----------
    def forward(self, texts: List[str], user_labels: torch.Tensor) -> torch.Tensor:
        """texts: lista de strings (len=B)
           user_labels: tensor [B] con labels del usuario (0..5)
        """
        h_text = self.embedder.embed_batch(texts)              # [B, D]
        onehot = self._one_hot(user_labels.to(h_text.device), self.label_dim)  # [B, 6]
        onehot = self.feat_drop(onehot)                        # feature dropout (solo en train)
        x = torch.cat([h_text, onehot], dim=-1)                # [B, D+6]
        logits = self.classifier(x)                            # [B, 6]
        return logits

    @torch.inference_mode()
    def predict(self, texts: List[str], user_labels: torch.Tensor):
        self.eval()
        logits = self.forward(texts, user_labels)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs



