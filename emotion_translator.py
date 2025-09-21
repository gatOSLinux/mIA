import pandas as pd
from datasets import load_dataset
from googletrans import Translator
import time
import json
from pathlib import Path
import random

class EmotionDatasetTranslator:
    def __init__(self, output_dir="models/emotion_classifier/data"):
        """
        Traductor de dataset de emociones inglés-español para MIA
        
        Args:
            output_dir: Directorio donde guardar los datos procesados
        """
        self.translator = Translator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapeo de emociones inglés -> español
        self.emotion_mapping = {
            0: "tristeza",   # sadness
            1: "alegria",    # joy  
            2: "amor",       # love
            3: "ira",        # anger
            4: "miedo",      # fear
            5: "sorpresa"    # surprise
        }
        
        # Mapeo original para referencia
        self.original_labels = {
            0: "sadness",
            1: "joy", 
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        
    def download_dataset(self):
        """
        Descarga el dataset de emociones de HuggingFace
        """
        print("📥 Descargando dataset de emociones...")
        
        try:
            # Cargar dataset con split balanceado
            dataset = load_dataset("dair-ai/emotion", split="train")
            
            print(f"✅ Dataset descargado: {len(dataset)} ejemplos")
            print(f"📊 Distribución por emoción:")
            
            # Mostrar distribución
            emotion_counts = {}
            for item in dataset:
                emotion = self.original_labels[item['label']]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in emotion_counts.items():
                print(f"   {emotion}: {count} ejemplos")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error descargando dataset: {e}")
            raise
    
    def translate_batch(self, texts, max_retries=3, delay=1):
        """
        Traduce un lote de textos con reintentos y delay
        """
        translated = []
        
        for i, text in enumerate(texts):
            for attempt in range(max_retries):
                try:
                    # Agregar delay para evitar rate limiting
                    if i > 0:
                        time.sleep(delay)
                    
                    # Traducir texto
                    result = self.translator.translate(text, src='en', dest='es')
                    translated.append(result.text)
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"⚠️  Error traduciendo: '{text[:50]}...' - Usando original")
                        translated.append(text)  # Usar original si falla
                    else:
                        time.sleep(delay * 2)  # Esperar más tiempo en reintento
        
        return translated
    
    def translate_dataset(self, dataset, batch_size=50, sample_size=None):
        """
        Traduce todo el dataset al español por lotes
        """
        print("🔄 Iniciando traducción del dataset...")
        
        # Muestreo si se especifica (para pruebas rápidas)
        if sample_size:
            indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
            dataset = dataset.select(indices)
            print(f"📋 Usando muestra de {len(dataset)} ejemplos")
        
        translated_data = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch_end = min(batch_idx + batch_size, len(dataset))
            batch = dataset[batch_idx:batch_end]
            
            print(f"🔄 Traduciendo lote {batch_idx//batch_size + 1}/{total_batches} "
                  f"({batch_idx+1}-{batch_end}/{len(dataset)})")
            
            # Extraer textos del lote
            texts = [item['text'] if isinstance(item, dict) else batch['text'][i] 
                    for i, item in enumerate(batch) if isinstance(batch, list)] or batch['text']
            labels = [item['label'] if isinstance(item, dict) else batch['label'][i] 
                     for i, item in enumerate(batch) if isinstance(batch, list)] or batch['label']
            
            # Traducir lote
            translated_texts = self.translate_batch(texts)
            
            # Agregar a datos traducidos
            for text_es, label in zip(translated_texts, labels):
                translated_data.append({
                    'text': text_es,
                    'label': label,
                    'emotion': self.emotion_mapping[label]
                })
        
        print(f"✅ Traducción completada: {len(translated_data)} ejemplos procesados")
        return translated_data
    
    def clean_translations(self, data):
        """
        Limpia y valida las traducciones
        """
        print("🧹 Limpiando traducciones...")
        
        cleaned_data = []
        
        for item in data:
            text = item['text'].strip()
            
            # Filtros de calidad básicos
            if len(text) < 5:  # Muy corto
                continue
            if len(text) > 280:  # Muy largo (como tweets)
                text = text[:280]
            
            # Limpiar caracteres extraños comunes en traducciones
            text = text.replace('&quot;', '"')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            
            cleaned_data.append({
                'text': text,
                'label': item['label'],
                'emotion': item['emotion']
            })
        
        print(f"✅ Limpieza completada: {len(cleaned_data)} ejemplos válidos")
        return cleaned_data
    
    def save_dataset(self, data, filename="emotion_dataset_es.json"):
        """
        Guarda el dataset traducido
        """
        output_path = self.output_dir / filename
        
        # Crear estadísticas
        stats = {
            'total_examples': len(data),
            'emotions': {},
            'avg_text_length': sum(len(item['text']) for item in data) / len(data)
        }
        
        for item in data:
            emotion = item['emotion']
            stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1
        
        # Estructura final
        dataset_output = {
            'metadata': {
                'source': 'dair-ai/emotion (translated to Spanish)',
                'translated_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'purpose': 'MIA emotion classifier training',
                'statistics': stats
            },
            'data': data
        }
        
        # Guardar
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset guardado en: {output_path}")
        
        # Guardar también como CSV para fácil inspección
        csv_path = self.output_dir / filename.replace('.json', '.csv')
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"📊 CSV guardado en: {csv_path}")
        
        return output_path
    
    def show_samples(self, data, n=5):
        """
        Muestra ejemplos del dataset traducido
        """
        print(f"\n🔍 Mostrando {n} ejemplos traducidos:")
        print("=" * 60)
        
        for i, item in enumerate(random.sample(data, min(n, len(data)))):
            print(f"\nEjemplo {i+1}:")
            print(f"Texto: {item['text']}")
            print(f"Emoción: {item['emotion']} (label: {item['label']})")
            print("-" * 40)

def main():
    """
    Función principal para traducir el dataset
    """
    print("🤖 MIA Emotion Dataset Translator")
    print("=" * 50)
    
    # Crear traductor
    translator = EmotionDatasetTranslator()
    
    try:
        # Paso 1: Descargar dataset
        dataset = translator.download_dataset()
        
        # Paso 2: Traducir (usar sample_size=1000 para prueba rápida)
        # Para dataset completo, comentar sample_size
        translated_data = translator.translate_dataset(
            dataset, 
            batch_size=20,  # Lotes pequeños para evitar rate limiting
            sample_size=2000  # Usar 2000 ejemplos para comenzar
        )
        
        # Paso 3: Limpiar traducciones
        cleaned_data = translator.clean_translations(translated_data)
        
        # Paso 4: Guardar dataset
        output_path = translator.save_dataset(cleaned_data)
        
        # Paso 5: Mostrar ejemplos
        translator.show_samples(cleaned_data)
        
        print("\n🎉 ¡Dataset de emociones listo para entrenar!")
        print(f"📁 Ubicación: {output_path}")
        print("\n📋 Próximo paso: Entrenar el clasificador de emociones")
        
    except Exception as e:
        print(f"❌ Error en el proceso: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()