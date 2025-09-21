import os
import json
import gc
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents
import multiprocessing as mp
import ijson  # Para streaming JSON parsing

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
        Extrae todos los textos del archivo corpus_analyzed.json usando streaming
        para evitar cargar 5.8GB en memoria de una vez
        """
        print("🔍 Buscando archivo corpus_analyzed.json...")
        
        # Buscar específicamente el archivo corpus_analyzed.json
        corpus_file = self.data_path / "corpus_analyzed.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo {corpus_file}")
        
        print(f"📁 Encontrado archivo: {corpus_file}")
        print(f"📊 Tamaño del archivo: {corpus_file.stat().st_size / (1024**3):.2f} GB")
        
        # Generador que yielda textos usando streaming JSON
        def text_generator():
            print(f"📖 Procesando corpus_analyzed.json con streaming...")
            
            try:
                batch_texts = []
                processed_articles = 0
                
                # Usar ijson para streaming parsing
                with open(corpus_file, 'rb') as f:
                    # Parsear los artículos uno por uno sin cargar todo en memoria
                    articles_parser = ijson.items(f, 'articles.item')
                    
                    print("   🚀 Iniciando procesamiento streaming...")
                    
                    for i, article in enumerate(articles_parser):
                        # Extraer título y contenido
                        title = article.get('title', '').strip() if article.get('title') else ''
                        content = article.get('content', '').strip() if article.get('content') else ''
                        
                        # Combinar título y contenido si ambos existen
                        if title and content:
                            text = f"{title}. {content}"
                        elif content:
                            text = content
                        elif title:
                            text = title
                        else:
                            continue
                        
                        # Validar que el texto no esté vacío y tenga contenido útil
                        if len(text.strip()) > 20:  # Al menos 20 caracteres
                            batch_texts.append(text)
                            processed_articles += 1
                        
                        # Yield por batches para controlar memoria
                        if len(batch_texts) >= self.batch_size:
                            for text in batch_texts:
                                yield text
                            batch_texts = []
                            
                            # Mostrar progreso cada 500 artículos
                            if processed_articles % 500 == 0:
                                print(f"   📈 Procesados: {processed_articles:,} artículos")
                            
                            # Limpiar memoria periódicamente
                            if processed_articles % 2000 == 0:
                                gc.collect()
                    
                    # Yield textos restantes del último batch
                    for text in batch_texts:
                        yield text
                    
                    print(f"   ✅ Procesamiento completado:")
                    print(f"      - Artículos procesados: {processed_articles:,}")
                    print(f"      - Promedio de caracteres por texto: {sum(len(t) for t in batch_texts) // len(batch_texts) if batch_texts else 0}")
                
                # Limpiar memoria final
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error durante el streaming parsing: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        return text_generator()
    
    def create_tokenizer(self):
        """
        Crea tokenizador BPE optimizado para español con manejo correcto de acentos
        """
        print("🔧 Creando tokenizador BPE optimizado para español...")
        
        # Crear tokenizador BPE
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # NORMALIZACIÓN OPTIMIZADA PARA ESPAÑOL:
        # No usar NFD que separa acentos - mantener caracteres compuestos
        from tokenizers.normalizers import Sequence, Replace, Strip
        
        tokenizer.normalizer = Sequence([
            Strip(),  # Quitar espacios al inicio/final
            Replace(r'\s+', ' '),  # Múltiples espacios -> un espacio
            # NO normalizar acentos - mantenerlos como están
        ])
        
        # PRE-TOKENIZACIÓN OPTIMIZADA PARA ESPAÑOL:
        # Cambiar ByteLevel por Whitespace + Punctuation para mejor manejo de acentos
        from tokenizers.pre_tokenizers import Sequence as PreSequence, Whitespace, Punctuation
        
        tokenizer.pre_tokenizer = PreSequence([
            Whitespace(),  # Separar por espacios
            Punctuation(behavior="isolated")  # Separar puntuación manteniendo contexto
        ])
        
        # DECODIFICADOR OPTIMIZADO:
        # Usar WordPiece en lugar de ByteLevel para mejor reconstrucción
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        
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
        Prueba el tokenizador con ejemplos en español incluyendo acentos
        """
        print("\n🧪 Probando tokenizador optimizado para español...")
        
        test_texts = [
            "Hola, soy MIA, tu asistente virtual amigable.",
            "¿Cómo te sientes hoy? Estoy aquí para ayudarte.",
            "Me encanta hablar contigo sobre cualquier tema.",
            "¡Qué emocionante poder conversar en español!",
            "Puedo ayudarte con información, consejos y compañía.",
            "La niña está en España comiendo paella.",
            "Acentos: á, é, í, ó, ú, ü y la letra ñ."
        ]
        
        all_perfect = True
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            
            # Verificar si la decodificación es perfecta
            is_perfect = text == decoded
            status = "✅ PERFECTO" if is_perfect else "❌ ERROR"
            
            if not is_perfect:
                all_perfect = False
            
            print(f"\n📝 Texto: {text}")
            print(f"🔢 Tokens: {encoded.tokens[:15]}{'...' if len(encoded.tokens) > 15 else ''}")
            print(f"📊 Cantidad: {len(encoded.ids)} tokens")
            print(f"🔄 Decodificado: {decoded}")
            print(f"🎯 Estado: {status}")
        
        print(f"\n{'🎉 ¡TOKENIZADOR PERFECTO!' if all_perfect else '⚠️  NECESITA AJUSTES'}")
        return all_perfect
    
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
    
    # Configurar rutas (corregidas para tu sistema)
    data_path = "/home/Downloads/Dataset-MIA/data_categorizada"  # ← Ruta absoluta corregida
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
    
    # Crear entrenador con configuración optimizada para archivos grandes
    trainer = MIATokenizerTrainer(
        data_path=data_path,
        output_path=output_path,
        vocab_size=40000,  # Puedes ajustar entre 32k-50k
        batch_size=100     # Muy reducido para streaming de 5.8GB
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