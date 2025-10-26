"""
=== MIA · Script de Inferencia del Clasificador de Emociones ===
Usa el modelo entrenado para hacer predicciones en tiempo real
"""

import torch
from emotion_classifier_model import EmotionClassifier
from pathlib import Path
import numpy as np


class EmotionPredictor:
    """
    Clase para cargar y usar el modelo entrenado
    """
    def __init__(self, model_path: str, device: torch.device = None):
        """
        Args:
            model_path: ruta al directorio del modelo guardado
            device: dispositivo para inferencia
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear modelo
        self.model = EmotionClassifier(
            model_name="dccuchile/bert-base-spanish-wwm-cased",
            emb_dim=300,
            max_length=128,
            hidden1=128,
            hidden2=64,
            num_classes=6,
            dropout=0.3,
            device=self.device
        )
        
        # Cargar pesos
        checkpoint_path = Path(model_path) / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Modelo cargado desde: {checkpoint_path}")
        print(f"  Época de entrenamiento: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"  Device: {self.device}")
    
    def predict_single(self, text: str, return_all_probs: bool = False):
        """
        Predice la emoción de un texto individual
        
        Args:
            text: texto en español
            return_all_probs: si True, retorna todas las probabilidades
        
        Returns:
            Si return_all_probs=False: (emocion, confianza)
            Si return_all_probs=True: (emocion, confianza, dict_todas_probs)
        """
        emotion, probs = self.model.predict_single(text, return_probs=True)
        confidence = probs.max() * 100
        
        if return_all_probs:
            all_probs = {
                label: float(prob) * 100
                for label, prob in zip(self.model.label_map.values(), probs)
            }
            return emotion, confidence, all_probs
        
        return emotion, confidence
    
    def predict_batch(self, texts: list, return_all_probs: bool = False):
        """
        Predice emociones para un lote de textos
        
        Args:
            texts: lista de textos
            return_all_probs: si True, retorna todas las probabilidades
        
        Returns:
            Lista de tuplas con resultados
        """
        emotions, probs = self.model.predict(texts, return_probs=True)
        
        results = []
        for i, (emotion, prob_array) in enumerate(zip(emotions, probs)):
            confidence = prob_array.max() * 100
            
            if return_all_probs:
                all_probs = {
                    label: float(p) * 100
                    for label, p in zip(self.model.label_map.values(), prob_array)
                }
                results.append((emotion, confidence, all_probs))
            else:
                results.append((emotion, confidence))
        
        return results
    
    def analyze_text(self, text: str):
        """
        Análisis detallado de un texto
        """
        emotion, confidence, all_probs = self.predict_single(text, return_all_probs=True)
        
        # Ordenar probabilidades
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        print("="*60)
        print("ANÁLISIS DE EMOCIÓN")
        print("="*60)
        print(f"\nTexto: '{text}'")
        print(f"\n🎯 Emoción predicha: {emotion.upper()}")
        print(f"💪 Confianza: {confidence:.2f}%")
        print(f"\n📊 Distribución de probabilidades:")
        print("-"*60)
        for emo, prob in sorted_probs:
            bar = "█" * int(prob / 2)
            print(f"  {emo:12s}: {prob:5.2f}% {bar}")
        print("="*60)


# ==================== EJEMPLOS DE USO ====================
def demo_inferencia():
    """
    Ejemplos de cómo usar el predictor
    """
    print("\n" + "="*60)
    print("DEMO: Inferencia con Clasificador de Emociones MIA")
    print("="*60 + "\n")
    
    # Cargar modelo
    predictor = EmotionPredictor(
        model_path="models/emotion_classifier"
    )
    
    print("\n" + "="*60)
    print("EJEMPLO 1: Predicción individual simple")
    print("="*60)
    
    texto = "¡Estoy muy feliz con los resultados del proyecto!"
    emocion, confianza = predictor.predict_single(texto)
    print(f"\nTexto: {texto}")
    print(f"Emoción: {emocion}")
    print(f"Confianza: {confianza:.2f}%")
    
    print("\n" + "="*60)
    print("EJEMPLO 2: Análisis detallado")
    print("="*60)
    
    predictor.analyze_text("Me da mucho miedo pensar en el futuro")
    
    print("\n" + "="*60)
    print("EJEMPLO 3: Predicción por lotes")
    print("="*60)
    
    textos_prueba = [
        "Te amo muchísimo, eres lo mejor que me ha pasado",
        "Estoy muy enojado con esta situación injusta",
        "Me siento triste y solo en estos días",
        "¡Qué sorpresa tan increíble, no me lo esperaba!",
        "Tengo mucho miedo de fracasar en el examen",
        "Estoy muy contento, todo salió perfecto"
    ]
    
    resultados = predictor.predict_batch(textos_prueba, return_all_probs=False)
    
    print("\nResultados del batch:")
    print("-"*60)
    for texto, (emocion, confianza) in zip(textos_prueba, resultados):
        print(f"\n📝 {texto}")
        print(f"   → {emocion.upper()} ({confianza:.1f}%)")
    
    print("\n" + "="*60)
    print("EJEMPLO 4: Modo interactivo")
    print("="*60)
    print("\nEscribe un texto para analizar su emoción (o 'salir' para terminar):")
    
    while True:
        texto = input("\n> ")
        if texto.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego! 👋")
            break
        
        if texto.strip():
            predictor.analyze_text(texto)


# ==================== INTEGRACIÓN CON MIA ====================
class MIAEmotionModule:
    """
    Módulo de emociones para integrar en el agente MIA
    """
    def __init__(self, model_path: str = "models/emotion_classifier"):
        """
        Inicializa el módulo de emociones para MIA
        """
        self.predictor = EmotionPredictor(model_path)
        print("✓ Módulo de emociones inicializado para MIA")
    
    def get_emotion(self, user_input: str) -> dict:
        """
        Obtiene la emoción del input del usuario
        
        Args:
            user_input: texto del usuario
        
        Returns:
            dict con: emotion, confidence, all_probabilities
        """
        emotion, confidence, all_probs = self.predictor.predict_single(
            user_input,
            return_all_probs=True
        )
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': all_probs,
            'user_input': user_input
        }
    
    def process_conversation_turn(self, user_input: str) -> dict:
        """
        Procesa un turno de conversación
        Retorna información emocional para el Gestor de Diálogo
        """
        emotion_data = self.get_emotion(user_input)
        
        # Lógica adicional para MIA
        if emotion_data['confidence'] > 80:
            certainty = "high"
        elif emotion_data['confidence'] > 60:
            certainty = "medium"
        else:
            certainty = "low"
        
        return {
            **emotion_data,
            'certainty': certainty,
            'requires_empathetic_response': emotion_data['emotion'] in ['tristeza', 'miedo', 'ira']
        }


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mia":
        # Modo MIA
        print("\n🤖 Inicializando módulo de emociones para MIA...")
        mia_emotion = MIAEmotionModule()
        
        # Test del módulo
        test_input = "Estoy muy triste porque perdí mi trabajo"
        result = mia_emotion.process_conversation_turn(test_input)
        
        print("\n" + "="*60)
        print("TEST: Procesamiento de turno de conversación")
        print("="*60)
        print(f"\nInput: {test_input}")
        print(f"\nResultado:")
        for key, value in result.items():
            if key != 'probabilities':
                print(f"  {key}: {value}")
        print("\nProbabilidades:")
        for emo, prob in result['probabilities'].items():
            print(f"  {emo}: {prob:.2f}%")
    else:
        # Modo demo
        demo_inferencia()