# -*- coding: utf-8 -*-
"""
Inferencia para MIA (RustyLinux/MiaMotion)
- Carga el modelo desde archivos locales si existen (best_model.pt, config.json).
- Si no están en local, los descarga desde el Hugging Face Hub.
- Expone `predict(text: str)` para usar desde scripts, Spaces (Gradio) o servicios.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # si no está instalado, funcionará en local con archivos presentes

from emotion_classifier_model import EmotionClassifier

# ---------------- Config ----------------
REPO_ID = "RustyLinux/MiaMotion"  # tu repo en el Hub

LOCAL_CKPT = Path("best_model.pt")
LOCAL_CFG  = Path("config.json")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None  # cache global


def _resolve_paths() -> (str, str):
    """
    Retorna (ckpt_path, cfg_path). Busca primero local, si no, descarga del Hub.
    """
    if LOCAL_CKPT.exists() and LOCAL_CFG.exists():
        return str(LOCAL_CKPT.resolve()), str(LOCAL_CFG.resolve())

    if hf_hub_download is None:
        raise RuntimeError(
            "No se encontraron 'best_model.pt' y 'config.json' en local, "
            "y 'huggingface_hub' no está instalado para descargarlos."
        )

    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pt")
    cfg_path  = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    return ckpt_path, cfg_path


def _load_model() -> EmotionClassifier:
    global _model
    if _model is not None:
        return _model

    ckpt_path, cfg_path = _resolve_paths()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = EmotionClassifier(
        model_name=cfg.get("base_model_id", "dccuchile/bert-base-spanish-wwm-cased"),
        max_length=cfg.get("max_length", 128),
        hidden1=cfg.get("hidden1", 128),
        hidden2=cfg.get("hidden2", 64),
        num_classes=cfg.get("num_classes", 6),
        dropout=cfg.get("dropout", 0.3),
        device=_device,
        pretrained_encoder=cfg.get("pretrained_encoder", "beto"),
    )

    state = torch.load(ckpt_path, map_location=_device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # por si guardaste el state_dict directo
        model.load_state_dict(state)

    model.eval()
    _model = model
    return _model


def predict(text: str, return_probs: bool = False) -> Any:
    """
    Predice la emoción para un texto.
    - return_probs=True: devuelve (label:str, probs:list[float]) en el orden de model.label_map
    - return_probs=False: devuelve solo label:str
    """
    model = _load_model()
    if return_probs:
        label, probs = model.predict_single(text, return_probs=True)
        return label, probs.tolist()
    return model.predict_single(text)


if __name__ == "__main__":
    # Pruebas rápidas
    print(predict("Estoy muy contento con los resultados", return_probs=True))
    print(predict("Tengo miedo de lo que pueda pasar", return_probs=True))
