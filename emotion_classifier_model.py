"""
=== MIA · Clasificador de Emociones (TextEmbedder + MLP) ===
Arquitectura completa para clasificación de 6 emociones en español
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoTokenizer


# ==================== MÓDULO 1: TextEmbedder (tu código) ====================
class TextEmbedder(nn.Module):
    """
    Módulo de Embedding para el Agente (MIA).
    - Entrada: texto(s) en español
    - Salida: vector(es) [E] por oración para alimentar el Clasificador de emociones
    """
    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",  # BETO
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

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Para procesamiento en lote:
        texts -> matriz [B, E]
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"].to(self.device)          # [B, T]
        attention_mask = batch["attention_mask"].to(self.device) # [B, T]

        embeds = self.embedding(input_ids)                      # [B, T, E]
        mask = attention_mask.bool()                            # [B, T]
        
        if self.cls_id is not None:
            mask = mask & (input_ids != self.cls_id)
        if self.sep_id is not None:
            mask = mask & (input_ids != self.sep_id)

        mask_f = mask.unsqueeze(-1).float()                     # [B, T, 1]
        summed = (embeds * mask_f).sum(dim=1)                   # [B, E]
        counts = mask_f.sum(dim=1).clamp(min=1.0)               # [B, 1]
        sentence_vecs = summed / counts                         # [B, E]
        return sentence_vecs

    def embed_sentence(self, text: str) -> torch.Tensor:
        """
        Para uso en tiempo real:
        text -> vector [E]
        """
        return self.embed_batch([text])[0]


# ==================== MÓDULO 2: MLP Classifier ====================
class MLPClassifier(nn.Module):
    """
    Red Neuronal Feedforward para clasificación de emociones.
    
    Arquitectura:
    Input (300) → Dense(128) + ReLU + Dropout(0.3) 
                → Dense(64) + ReLU + Dropout(0.3) 
                → Dense(6) + Softmax
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
        
        # Capa oculta 1
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Capa oculta 2
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Capa de salida
        self.fc3 = nn.Linear(hidden2, num_classes)
        # Nota: No incluimos Softmax aquí porque CrossEntropyLoss lo hace internamente
    
    def forward(self, x):
        """
        Forward pass
        x: [B, input_dim] tensor de embeddings
        returns: [B, num_classes] logits
        """
        # Capa 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Capa 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Capa de salida
        x = self.fc3(x)
        
        return x  # Retorna logits (sin softmax)


# ==================== MÓDULO 3: Modelo Completo ====================
class EmotionClassifier(nn.Module):
    """
    Modelo completo que integra TextEmbedder + MLPClassifier
    
    Flujo:
    Texto → TextEmbedder (tokenization + embedding + pooling) → MLP → Logits
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
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Componente 1: Embedder (entrenable)
        self.embedder = TextEmbedder(
            model_name=model_name,
            emb_dim=emb_dim,
            max_length=max_length,
            device=self.device
        )
        
        # Componente 2: MLP Classifier (entrenable)
        self.classifier = MLPClassifier(
            input_dim=emb_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.to(self.device)
        
        # Mapeo de etiquetas a emociones
        self.label_map = {
            0: "tristeza",
            1: "alegría",
            2: "amor",
            3: "ira",
            4: "miedo",
            5: "sorpresa"
        }
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass completo
        texts: lista de strings
        returns: [B, num_classes] logits
        """
        # Paso 1: Obtener embeddings
        embeddings = self.embedder.embed_batch(texts)  # [B, emb_dim]
        
        # Paso 2: Clasificar
        logits = self.classifier(embeddings)  # [B, num_classes]
        
        return logits
    
    def predict(self, texts: List[str], return_probs: bool = False):
        """
        Predicción con conversión a etiquetas legibles
        
        Args:
            texts: lista de textos
            return_probs: si True, retorna también las probabilidades
        
        Returns:
            Si return_probs=False: lista de emociones predichas
            Si return_probs=True: tupla (emociones, probabilidades)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            # Convertir a nombres de emociones
            emotions = [self.label_map[pred.item()] for pred in predictions]
            
            if return_probs:
                return emotions, probs.cpu().numpy()
            return emotions
    
    def predict_single(self, text: str, return_probs: bool = False):
        """
        Predicción para un solo texto
        """
        result = self.predict([text], return_probs=return_probs)
        
        if return_probs:
            emotions, probs = result
            return emotions[0], probs[0]
        return result[0]


# ==================== EJEMPLO DE USO ====================
if __name__ == "__main__":
    print("="*60)
    print("Inicializando Clasificador de Emociones MIA")
    print("="*60)
    
    # Crear el modelo completo
    model = EmotionClassifier(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        emb_dim=300,
        max_length=128,
        hidden1=128,
        hidden2=64,
        num_classes=6,
        dropout=0.3
    )
    
    print(f"\nModelo creado en dispositivo: {model.device}")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test con textos de ejemplo
    print("\n" + "="*60)
    print("TEST: Forward pass con batch de textos")
    print("="*60)
    
    textos_prueba = [
        "Me siento muy feliz hoy, gracias por tu ayuda",
        "Estoy triste y decepcionado",
        "Te amo con todo mi corazón",
        "Esto me da mucha rabia y enojo",
        "Tengo miedo de lo que pueda pasar",
        "¡Qué sorpresa tan increíble!"
    ]
    
    # Forward pass (logits sin entrenar)
    logits = model(textos_prueba)
    print(f"\nLogits shape: {logits.shape}")  # [6, 6]
    print(f"Logits (sin entrenar):\n{logits}")
    
    # Predicción con softmax
    print("\n" + "="*60)
    print("TEST: Predicciones con softmax")
    print("="*60)
    
    emociones, probs = model.predict(textos_prueba, return_probs=True)
    
    for i, (texto, emocion, prob) in enumerate(zip(textos_prueba, emociones, probs)):
        print(f"\n{i+1}. Texto: {texto}")
        print(f"   Emoción predicha: {emocion}")
        print(f"   Probabilidades: {dict(zip(model.label_map.values(), prob.round(3)))}")
    
    # Test predicción individual
    print("\n" + "="*60)
    print("TEST: Predicción individual")
    print("="*60)
    
    texto_single = "Estoy muy contento con los resultados"
    emocion_single = model.predict_single(texto_single)
    print(f"\nTexto: {texto_single}")
    print(f"Emoción: {emocion_single}")
    
    print("\n" + "="*60)
    print("Modelo listo para entrenamiento!")
    print("="*60)