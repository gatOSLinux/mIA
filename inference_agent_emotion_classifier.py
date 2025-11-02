# -*- coding: utf-8 -*-
"""
Inferencia para el AgentEmotionPredictClassifier (MIA · segunda red)
- Busca 'best_model.pt' y 'config_agent.json' en local; si no están y hay
  huggingface_hub instalado, los descarga del repo indicado.
- La config DEBE incluir, como mínimo:
    {
      "base_model_id": "dccuchile/bert-base-spanish-wwm-cased",
      "max_length": 128,
      "hidden1": 256,
      "hidden2": 64,
      "num_classes": 2,
      "dropout": 0.4,
      "label_feature_dropout": 0.5,
      "pretrained_encoder": "beto",
      "present_classes": [1, 2],          # ids originales (0..5) presentes en train
      "class_names": ["alegría","amor"]   # nombres en el mismo orden del mapeo 0..K-1
    }

- Uso:
    from inference_agent_emotion import predict
    y = predict("No me siento bien", user_label=0)  # 0..5 (tristeza..sorpresa)
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import torch

# Opcional: descarga desde HF si no hay archivos locales
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

from agent_emotion_predict_classifier import AgentEmotionPredictClassifier

# ---------------- Config ----------------
REPO_ID = "RustyLinux/MiaPredict"  # cambia por tu repo si usas el Hub

LOCAL_CKPT = Path("best_model_agent.pt")           # checkpoint de la segunda red
LOCAL_CFG  = Path("config_agent.json")       # config de la segunda red

# Mapa global de emociones (usuario y también nombres canónicos)
EMOTION_ID2NAME = {
    0: "tristeza",
    1: "alegría",
    2: "amor",
    3: "ira",
    4: "miedo",
    5: "sorpresa",
}
EMOTION_NAME2ID = {v: k for k, v in EMOTION_ID2NAME.items()}

_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[AgentEmotionPredictClassifier] = None
_cfg: Optional[Dict[str, Any]] = None
_label_map_fwd: Optional[Dict[int, int]] = None   # original_id -> idx(0..K-1) usado en entrenamiento
_label_map_inv: Optional[Dict[int, int]] = None   # idx(0..K-1) -> original_id (para devolver nombre global)


# ---------------- Utilidades internas ----------------
def _resolve_paths() -> Tuple[str, str]:
    """
    Retorna (ckpt_path, cfg_path). Prefiere local; si no, intenta descarga HF.
    """
    if LOCAL_CKPT.exists() and LOCAL_CFG.exists():
        print("✅ Cargando archivos desde local.")
        return str(LOCAL_CKPT.resolve()), str(LOCAL_CFG.resolve())

    if hf_hub_download is None:
        raise RuntimeError(
            "No se encontraron 'best_model_agent.pt' y 'config_agent.json' en local, "
            "y 'huggingface_hub' no está instalado para descargarlos."
        )

    print("⬇️  Descargando archivos desde Hugging Face Hub...")
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="best_model_agent.pt")
    cfg_path  = hf_hub_download(repo_id=REPO_ID, filename="config_agent.json")
    return ckpt_path, cfg_path


def _prepare_label_maps(cfg: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Construye los mapeos entre ids originales (0..5) y los índices 0..K-1 usados por la head.
    """
    present = cfg.get("present_classes", None)
    if not present:
        # Por compatibilidad: si no viene, asumimos [0..num_classes-1], pero se recomienda guardarlo.
        k = int(cfg.get("num_classes", 2))
        present = list(range(k))
    present = list(sorted(int(x) for x in present))
    fwd = {orig: i for i, orig in enumerate(present)}
    inv = {i: orig for orig, i in fwd.items()}
    return fwd, inv


def _load_config(cfg_path: str) -> Dict[str, Any]:
    global _cfg, _label_map_fwd, _label_map_inv
    if _cfg is not None:
        return _cfg
    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
    _label_map_fwd, _label_map_inv = _prepare_label_maps(_cfg)
    return _cfg


def _build_model(cfg: Dict[str, Any]) -> AgentEmotionPredictClassifier:
    model = AgentEmotionPredictClassifier(
        model_name=cfg.get("base_model_id", "dccuchile/bert-base-spanish-wwm-cased"),
        pretrained_encoder=cfg.get("pretrained_encoder", "beto"),
        emb_dim=cfg.get("emb_dim", 300),
        max_length=cfg.get("max_length", 128),
        hidden1=cfg.get("hidden1", 256),
        hidden2=cfg.get("hidden2", 64),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.4),
        label_feature_dropout=cfg.get("label_feature_dropout", 0.0),  # en inferencia no se usa
        device=_device,
    )
    # aseguramos eval()
    model.eval()
    return model


def _load_model() -> AgentEmotionPredictClassifier:
    global _model
    if _model is not None:
        return _model

    ckpt_path, cfg_path = _resolve_paths()
    cfg = _load_config(cfg_path)

    model = _build_model(cfg)

    state = torch.load(ckpt_path, map_location=_device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    _model = model
    print(f"✅ Modelo cargado en {_device} | num_classes={cfg.get('num_classes')} | "
          f"present_classes={cfg.get('present_classes')}")
    return _model


def _coerce_user_label(label: Union[int, str]) -> int:
    """
    Convierte un label de usuario a id 0..5.
    - Si llega string ("alegría"), lo mapea.
    - Valida rango si llega int.
    """
    if isinstance(label, str):
        label = label.strip().lower()
        if label not in EMOTION_NAME2ID:
            raise ValueError(f"Label de usuario desconocido: {label}. Esperado uno de {list(EMOTION_NAME2ID.keys())}")
        return EMOTION_NAME2ID[label]
    if isinstance(label, int):
        if label < 0 or label > 5:
            raise ValueError("El user_label debe estar en 0..5.")
        return label
    raise TypeError("user_label debe ser int (0..5) o str (nombre de emoción).")


def _map_agent_idx_to_original(idx: int) -> int:
    """
    Convierte el índice 0..K-1 (head) al id original 0..5 para reportar el nombre global.
    """
    if _label_map_inv is None:
        raise RuntimeError("Mapeos de etiquetas no inicializados.")
    return _label_map_inv[int(idx)]


def _agent_class_names() -> List[str]:
    """
    Nombres de clases del agente en el mismo orden que la head (0..K-1).
    """
    if _cfg is None:
        raise RuntimeError("Config no cargada.")
    names = _cfg.get("class_names", None)
    if names:
        return list(names)
    # fallback: usar nombres globales segun present_classes
    present = sorted(_cfg.get("present_classes", []))
    return [EMOTION_ID2NAME[p] for p in present]


# ---------------- API de inferencia ----------------
@torch.inference_mode()
def predict(text: str, user_label: Union[int, str], return_probs: bool = False) -> Any:
    """
    Predice la emoción CON LA QUE DEBE RESPONDER EL AGENTE.
    Args:
        text: str
        user_label: int(0..5) o nombre ("tristeza", "alegría", "amor", "ira", "miedo", "sorpresa")
        return_probs: si True devuelve (pred_name, probs_dict)

    Returns:
        - Si return_probs=False: str con el nombre de la emoción objetivo del agente (en nombres globales 0..5).
        - Si return_probs=True: (pred_name:str, probs:Dict[str,float]) usando los nombres en orden de la head.
    """
    model = _load_model()
    cfg = _cfg  # ya cargada
    assert cfg is not None

    # 1) preparar entrada
    u = _coerce_user_label(user_label)
    user_tensor = torch.tensor([u], dtype=torch.long, device=_device)
    texts = [text]

    # 2) forward
    logits = model(texts, user_tensor)              # [1, K]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())

    # 3) mapear idx(0..K-1) -> id original (0..5) y nombre canónico
    orig_id = _map_agent_idx_to_original(pred_idx)
    pred_name = EMOTION_ID2NAME[orig_id]

    if not return_probs:
        return pred_name

    # nombres amistosos en el orden de la head
    names_head = _agent_class_names()
    probs_dict = {names_head[i]: float(probs[i]) for i in range(len(names_head))}
    return pred_name, probs_dict


@torch.inference_mode()
def predict_batch(texts: List[str], user_labels: List[Union[int, str]], return_probs: bool = False):
    """
    Batch de inferencia.
    - user_labels: lista paralela a texts con ids (0..5) o nombres de emoción.
    """
    if len(texts) != len(user_labels):
        raise ValueError("texts y user_labels deben tener la misma longitud.")
    model = _load_model()

    # preparar
    u_ids = [ _coerce_user_label(u) for u in user_labels ]
    user_tensor = torch.tensor(u_ids, dtype=torch.long, device=_device)

    logits = model(texts, user_tensor)      # [B, K]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idxs = probs.argmax(axis=1)

    results = []
    names_head = _agent_class_names()

    for i, idx in enumerate(pred_idxs):
        orig_id = _map_agent_idx_to_original(int(idx))
        pred_name = EMOTION_ID2NAME[orig_id]
        if return_probs:
            pvec = probs[i]
            probs_dict = {names_head[j]: float(pvec[j]) for j in range(len(names_head))}
            results.append((pred_name, probs_dict))
        else:
            results.append(pred_name)
    return results


# ---------------- CLI rápido ----------------
if __name__ == "__main__":
    # Ejemplos rápidos
    txts = [
        "Tuve ese tipo de sentimiento pero lo ignoré",
        "Estoy muy feliz con la noticia",
        "Me molesta lo que pasó",
    ]
    # user_label puede ser int o str
    for t, ulab in zip(txts, [0, "alegría", "ira"]):
        out = predict(t, user_label=ulab, return_probs=True)
        print(f"\nTexto: {t}\nUser label: {ulab}\nPredicción agente: {out[0]}\nProbs: {out[1]}")
