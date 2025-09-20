import os
import json
import gc
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents
import multiprocessing as mp

class MIATokenizerTrainer:
    def __init__(self, data_path, output_path, vocab_size=40000, batch_size=1000):
        """
        Entrenador de tokenizador BPE para MIA
        
        Args:
            data_path: Ruta a Downloads/Dataset-MIA/data_categorized
            output_path: Ruta donde guardar el tokenizador (ej: MIA/models/tokenizer/)
            vocab_size: Tamaño del vocabulario (32k-50k)
            batch_size: Tamaño de batch para procesar en memoria
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        # Tokens especiales para MIA (asistente amigable)
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "[MIA]", "[USER]", "[EMO]", "[MEM]",
            "[HAPPY]", "[SAD]", "[EXCITED]", "[CALM]", "[LOVE]", "[CARE]"
        ]
        
        # Crear directorios si no existen
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def extract_texts_from_json(self):
        """
        Extrae todos los textos de los archivos JSON procesando por batches
        para evitar problemas de memoria
        """
        print("🔍 Buscando archivos JSON...")
        json_files = list(self.data_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No se encontraron archivos JSON en {self.data_path}")
        
        print(f"📁 Encontrados {len(json_files)} archivos JSON")
        
        # Generador que yielda textos por batches
        def text_generator():
            for json_file in json_files:
                print(f"📖 Procesando: {json_file.name}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extraer artículos
                    articles = data.get('articles', [])
                    print(f"   └─ {len(articles)} artículos encontrados")
                    
                    batch_texts = []
                    for i, article in enumerate(articles):
                        # Extraer título y contenido
                        title = article.get('title', '').strip()
                        content = article.get('content', '').strip()
                        
                        # Combinar título y contenido si ambos existen
                        if title and content:
                            text = f"{title}. {content}"
                        elif content:
                            text = content
                        elif title:
                            text = title
                        else:
                            continue
                        
                        batch_texts.append(text)
                        
                        # Yield por batches para controlar memoria
                        if len(batch_texts) >= self.batch_size:
                            for text in batch_texts:
                                yield text
                            batch_texts = []
                            
                            # Limpiar memoria
                            if (i + 1) % (self.batch_size * 2) == 0:
                                gc.collect()
                    
                    # Yield textos restantes del batch
                    for text in batch_texts:
                        yield text
                    
                    # Limpiar memoria después de cada archivo
                    del data, articles
                    gc.collect()
                    
                except Exception as e:
                    print(f"❌ Error procesando {json_file.name}: {e}")
                    continue
        
        return text_generator()
    
    def create_tokenizer(self):
        """
        Crea y configura el tokenizador BPE
        """
        print("🔧 Creando tokenizador BPE...")
        
        # Crear tokenizador BPE
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Normalización: mantener acentos para español
        tokenizer.normalizer = NFD()
        
        # Pre-tokenización: dividir por espacios y puntuación
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Decodificador
        tokenizer.decoder = decoders.ByteLevel()
        
        # Post-procesador: añadir tokens especiales
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.special_tokens.index("[CLS]")),
                ("[SEP]", self.special_tokens.index("[SEP]")),
            ],
        )
        
        return tokenizer
    
    def train_tokenizer(self):
        """
        Entrena el tokenizador con el corpus
        """
        print("🚀 Iniciando entrenamiento del tokenizador...")
        print(f"   📊 Vocabulario objetivo: {self.vocab_size:,} tokens")
        print(f"   🔤 Tokens especiales: {len(self.special_tokens)}")
        
        # Crear tokenizador
        tokenizer = self.create_tokenizer()
        
        # Configurar entrenador
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            continuing_subword_prefix="##",
            show_progress=True
        )
        
        # Extraer textos del corpus
        text_generator = self.extract_texts_from_json()
        
        # Entrenar
        print("🎓 Entrenando tokenizador...")
        tokenizer.train_from_iterator(text_generator, trainer=trainer)
        
        return tokenizer
    
    def save_tokenizer(self, tokenizer):
        """
        Guarda el tokenizador entrenado
        """
        print("💾 Guardando tokenizador...")
        
        # Guardar archivo principal del tokenizador
        tokenizer_path = self.output_path / "mia_tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        
        # Guardar configuración adicional
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "model_type": "BPE",
            "trained_on": "Spanish Wikipedia + Cultural Content",
            "purpose": "MIA Virtual Assistant"
        }
        
        config_path = self.output_path / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Tokenizador guardado en: {tokenizer_path}")
        print(f"✅ Configuración guardada en: {config_path}")
        
        return tokenizer_path
    
    def test_tokenizer(self, tokenizer):
        """
        Prueba el tokenizador con ejemplos
        """
        print("\n🧪 Probando tokenizador...")
        
        test_texts = [
            "Hola, soy MIA, tu asistente virtual amigable.",
            "¿Cómo te sientes hoy? Estoy aquí para ayudarte.",
            "Me encanta hablar contigo sobre cualquier tema.",
            "¡Qué emocionante poder conversar en español!",
            "Puedo ayudarte con información, consejos y compañía."
        ]
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            
            print(f"\n📝 Texto: {text}")
            print(f"🔢 Tokens: {encoded.tokens[:10]}{'...' if len(encoded.tokens) > 10 else ''}")
            print(f"📊 Cantidad: {len(encoded.ids)} tokens")
            print(f"🔄 Decodificado: {decoded}")
    
    def get_vocab_stats(self, tokenizer):
        """
        Obtiene estadísticas del vocabulario
        """
        vocab = tokenizer.get_vocab()
        
        print(f"\n📈 Estadísticas del vocabulario:")
        print(f"   Total de tokens: {len(vocab):,}")
        print(f"   Tokens especiales: {len(self.special_tokens)}")
        print(f"   Tokens de contenido: {len(vocab) - len(self.special_tokens):,}")
        
        # Mostrar algunos tokens ejemplo
        sample_tokens = list(vocab.keys())[:20]
        print(f"   Muestra de tokens: {sample_tokens}")

def main():
    """
    Función principal para entrenar el tokenizador
    """
    print("🤖 MIA Tokenizer Trainer - Entrenador de Tokenizador BPE")
    print("=" * 60)
    
    # Configurar rutas (ajusta según tu sistema)
    data_path = "/home/Downloads/Dataset-MIA/data_categorizada"
    output_path = "mIA/models/tokenizer"  # Se creará automáticamente
    
    # Verificar que existe la carpeta de datos
    if not Path(data_path).exists():
        print(f"❌ Error: No se encuentra la carpeta {data_path}")
        print("   Por favor, verifica la ruta a tus datos.")
        return
    
    # Crear entrenador
    trainer = MIATokenizerTrainer(
        data_path=data_path,
        output_path=output_path,
        vocab_size=40000,  # Puedes ajustar entre 32k-50k
        batch_size=500     # Reducido para tu RAM de 24GB
    )
    
    try:
        # Entrenar tokenizador
        tokenizer = trainer.train_tokenizer()
        
        # Guardar
        tokenizer_path = trainer.save_tokenizer(tokenizer)
        
        # Probar
        trainer.test_tokenizer(tokenizer)
        
        # Estadísticas
        trainer.get_vocab_stats(tokenizer)
        
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Tokenizador disponible en: {tokenizer_path}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()