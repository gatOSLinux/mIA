# mia_models_server.py
# Microservicio local que usa TUS módulos (sin transformers.pipeline)
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
# Opcional: fija carpeta de caché de HF (descarga una sola vez y listo)
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # <-- OCULTA GPU PARA SIEMPRE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HOME", os.path.abspath("./hf_cache"))
os.environ.setdefault("HF_HOME", os.path.abspath("./hf_cache"))

# Importa tus módulos tal como los tienes
# Asegúrate de que estos archivos estén en la MISMA carpeta:
#  - MiaMotion.py + emotion_classifier_model.py (+ best_model.pt, config.json o los descargará)
#  - MiaPredict.py + agent_emotion_predict_classifier.py (+ best_model_agent.pt, config_agent.json o los descargará)
from inference_emotion_classifier import predict as motion_predict
from inference_agent_emotion_classifier import predict as agent_predict

app = FastAPI(title="MIA Local Models", version="1.0.0")

# === Schemas ===
class SentimentIn(BaseModel):
    text: str
    return_probs: Optional[bool] = False

class MiaPredictIn(BaseModel):
    text: str
    sentimiento: str            # "tristeza","alegría","amor","ira","miedo","sorpresa"
    return_probs: Optional[bool] = False

# Normalización por si llega con mayúsculas/espacios
def norm_label(s: str) -> str:
    return s.strip().lower()

# Mapa global (por si necesitas validar/normalizar)
VALID_SENTIMENTS = {"tristeza","alegría","amor","ira","miedo","sorpresa"}

@app.post("/sentiment")
def sentiment_api(inp: SentimentIn):
    """
    Usa TU MiaMotion.predict(text, return_probs=False/True).
    Devuelve {'sentimiento': <str>} o {'sentimiento': <str>, 'probs': [...] } si pides probs.
    """
    if inp.return_probs:
        label, probs = motion_predict(inp.text, return_probs=True)
        return {"sentimiento": norm_label(label), "probs": probs}
    else:
        label = motion_predict(inp.text, return_probs=False)
        return {"sentimiento": norm_label(label)}

@app.post("/mia_predict")
def mia_predict_api(inp: MiaPredictIn):
    """
    Usa TU MiaPredict.predict(text, user_label=<id o nombre>, return_probs=False/True).
    - Aquí le pasamos el nombre de emoción del usuario (string), tal como lo espera tu código.
    Devuelve {'mia_emocion': <str>} o con 'probs' si pides probs.
    """
    sent = norm_label(inp.sentimiento)
    if sent not in VALID_SENTIMENTS:
        return {"error": f"sentimiento inválido: '{inp.sentimiento}'. Debe ser uno de {sorted(VALID_SENTIMENTS)}"}

    if inp.return_probs:
        pred_name, probs = agent_predict(inp.text, user_label=sent, return_probs=True)
        return {"mia_emocion": norm_label(pred_name), "probs": probs}
    else:
        pred_name = agent_predict(inp.text, user_label=sent, return_probs=False)
        return {"mia_emocion": norm_label(pred_name)}
