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
        Extrae todos los textos del archivo corpus_analyzed.json procesando por batches
        para evitar problemas de memoria
        """
        print("🔍 Buscando archivo corpus_analyzed.json...")
        
        # Buscar específicamente el archivo corpus_analyzed.json
        corpus_file = self.data_path / "corpus_analyzed.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo {corpus_file}")
        
        print(f"📁 Encontrado archivo: {corpus_file}")
        print(f"📊 Tamaño del archivo: {corpus_file.stat().st_size / (1024**3):.2f} GB")
        
        # Generador que yielda textos por batches
        def text_generator():
            print(f"📖 Procesando: corpus_analyzed.json")
            
            try:
                # Leer el archivo JSON completo
                print("   ⏳ Cargando archivo JSON en memoria...")
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print("   ✅ Archivo JSON cargado exitosamente")
                
                # Extraer metadata y artículos
                metadata = data.get('metadata', {})
                articles = data.get('articles', [])
                
                total_articles = len(articles)
                total_words = metadata.get('statistics', {}).get('total_words', 0)
                
                print(f"   📚 Total de artículos: {total_articles:,}")
                print(f"   📝 Total de palabras: {total_words:,}")
                print(f"   🔄 Procesando por batches de {self.batch_size}...")
                
                batch_texts = []
                processed_articles = 0
                
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
                    
                    # Validar que el texto no esté vacío
                    if len(text.strip()) > 10:  # Al menos 10 caracteres
                        batch_texts.append(text)
                        processed_articles += 1
                    
                    # Yield por batches para controlar memoria
                    if len(batch_texts) >= self.batch_size:
                        for text in batch_texts:
                            yield text
                        batch_texts = []
                        
                        # Mostrar progreso cada 1000 artículos
                        if (i + 1) % 1000 == 0:
                            progress = ((i + 1) / total_articles) * 100
                            print(f"   📈 Progreso: {progress:.1f}% ({i + 1:,}/{total_articles:,} artículos)")
                        
                        # Limpiar memoria periódicamente
                        if (i + 1) % (self.batch_size * 5) == 0:
                            gc.collect()
                
                # Yield textos restantes del último batch
                for text in batch_texts:
                    yield text
                
                print(f"   ✅ Procesamiento completado:")
                print(f"      - Artículos procesados: {processed_articles:,}/{total_articles:,}")
                print(f"      - Artículos válidos: {(processed_articles/total_articles)*100:.1f}%")
                
                # Limpiar memoria final
                del data, articles
                gc.collect()
                
            except json.JSONDecodeError as e:
                print(f"❌ Error al decodificar JSON: {e}")
                raise
            except MemoryError as e:
                print(f"❌ Error de memoria al cargar el archivo: {e}")
                print("   💡 Intenta reducir batch_size o liberar memoria del sistema")
                raise
            except Exception as e:
                print(f"❌ Error inesperado procesando corpus_analyzed.json: {e}")
                raise
        
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
    
    # Configurar rutas (ajustadas para tu estructura real)
    data_path = "/home/Downloads/Dataset-MIA/data_categorizada"  # ← Ruta corregida
    output_path = "MIA/models/tokenizer"  # Se creará automáticamente
    
    # Verificar que existe la carpeta de datos
    if not Path(data_path).exists():
        print(f"❌ Error: No se encuentra la carpeta {data_path}")
        print("   Por favor, verifica la ruta a tus datos.")
        return
    
    # Verificar que existe el archivo específico
    corpus_file = Path(data_path) / "corpus_analyzed.json"
    if not corpus_file.exists():
        print(f"❌ Error: No se encuentra el archivo {corpus_file}")
        print("   Por favor, verifica que el archivo corpus_analyzed.json existe.")
        return
    
    print(f"✅ Datos encontrados en: {data_path}")
    print(f"✅ Archivo corpus: {corpus_file}")
    
    # Crear entrenador
    trainer = MIATokenizerTrainer(
        data_path=data_path,
        output_path=output_path,
        vocab_size=40000,  # Puedes ajustar entre 32k-50k
        batch_size=200     # Reducido para manejar el archivo de 5.8GB
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