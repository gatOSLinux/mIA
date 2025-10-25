#!/usr/bin/env python3
"""
Analizador MIA con procesamiento streaming para archivos grandes
"""

import json
import re
import os
from collections import Counter, defaultdict
import argparse
from datetime import datetime
import numpy as np

class StreamingMIAAnalyzer:
    def __init__(self):
        # Categorías específicas para el entrenamiento de MIA
        self.categories = {
            'conversational': {
                'keywords': ['persona', 'gente', 'vida', 'familia', 'amigo', 'trabajo', 'estudio', 'sentir', 'pensar'],
                'description': 'Contenido útil para conversación natural'
            },
            'emotional': {
                'keywords': ['amor', 'alegría', 'tristeza', 'miedo', 'enojo', 'feliz', 'triste', 'emoción', 'sentimiento'],
                'description': 'Contenido con carga emocional para entrenamiento empático'
            },
            'educational': {
                'keywords': ['ciencia', 'historia', 'arte', 'cultura', 'conocimiento', 'aprender', 'explicar'],
                'description': 'Contenido informativo y educativo'
            },
            'biographical': {
                'keywords': ['nació', 'murió', 'vida', 'carrera', 'obra', 'escritor', 'artista', 'político'],
                'description': 'Biografías útiles para referencias contextuales'
            },
            'cultural': {
                'keywords': ['tradición', 'costumbre', 'festival', 'música', 'literatura', 'español', 'cultura'],
                'description': 'Contenido cultural hispano relevante'
            },
            'social': {
                'keywords': ['sociedad', 'comunidad', 'grupo', 'relación', 'comunicación', 'interacción'],
                'description': 'Contenido sobre dinámicas sociales'
            }
        }
        
        # Filtros relajados para archivos grandes
        self.quality_thresholds = {
            'min_words': 25,
            'max_words': 5000,
            'min_sentences': 2,
            'diversity_threshold': 0.2
        }
    
    def analyze_article(self, article):
        """Analiza un artículo y extrae métricas"""
        content = article.get('content', '')
        if not content:
            return None
            
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Métricas básicas
        metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'char_count': len(content),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
        
        # Filtros básicos
        if not self._passes_basic_filters(metrics, content):
            return None
        
        # Puntuación de calidad relajada
        quality_score = self._calculate_training_quality(metrics, content)
        
        # Categorización
        category_scores = self._categorize_content(content)
        primary_category = max(category_scores, key=category_scores.get) if category_scores else 'general'
        
        # Potencial conversacional
        conversation_potential = self._assess_mia_potential(content)
        
        return {
            'metrics': metrics,
            'quality_score': quality_score,
            'primary_category': primary_category,
            'category_scores': category_scores,
            'conversation_potential': conversation_potential,
            'training_priority': self._determine_training_priority(quality_score, category_scores, conversation_potential)
        }
    
    def _passes_basic_filters(self, metrics, content):
        """Filtros básicos relajados"""
        word_count = metrics['word_count']
        
        if word_count < self.quality_thresholds['min_words']:
            return False
        if word_count > self.quality_thresholds['max_words']:
            return False
        
        # Filtro de diversidad relajado
        if metrics['lexical_diversity'] < self.quality_thresholds['diversity_threshold']:
            return False
        
        return True
    
    def _calculate_training_quality(self, metrics, content):
        """Calcula calidad relajada"""
        score = 0.0
        word_count = metrics['word_count']
        
        # Longitud (peso: 0.3) - más permisivo
        if 50 <= word_count <= 1000:
            score += 0.3
        elif 25 <= word_count < 50 or 1000 < word_count <= 2000:
            score += 0.25
        elif word_count > 2000:
            score += 0.2
        
        # Diversidad léxica (peso: 0.25) - umbrales bajos
        diversity = metrics['lexical_diversity']
        if diversity >= 0.4:
            score += 0.25
        elif diversity >= 0.3:
            score += 0.2
        elif diversity >= 0.2:
            score += 0.15
        
        # Estructura (peso: 0.2)
        avg_length = metrics['avg_sentence_length']
        if 5 <= avg_length <= 30:
            score += 0.2
        elif avg_length > 30:
            score += 0.1
        
        # Contenido conversacional (peso: 0.15)
        conversational_markers = ['es', 'son', 'puede', 'debe', 'tiene', 'hace']
        if word_count > 0:
            marker_density = sum(1 for marker in conversational_markers if marker in content.lower()) / word_count
            score += 0.15 * min(marker_density * 100, 1.0)
        
        # Puntuación básica (peso: 0.1)
        if re.search(r'[.!?]', content):
            score += 0.1
        
        return min(score, 1.0)
    
    def _categorize_content(self, content):
        """Categorización simple"""
        content_lower = content.lower()
        words = content_lower.split()
        word_set = set(words)
        
        category_scores = {}
        for category, info in self.categories.items():
            matches = sum(1 for keyword in info['keywords'] if keyword in word_set)
            score = matches / len(words) * 1000 if words else 0
            category_scores[category] = score
        
        return category_scores
    
    def _assess_mia_potential(self, content):
        """Evaluación simple de potencial"""
        words = content.lower().split()
        if not words:
            return {'empathy_training': 0, 'knowledge_base': 0, 'dialogue_patterns': 0, 'emotional_context': 0}
        
        empathy_words = ['sentir', 'emoción', 'ayuda', 'apoyo']
        knowledge_words = ['es', 'son', 'significa', 'consiste']
        dialogue_words = ['dice', 'habla', 'explica', 'cuenta']
        emotion_words = ['alegre', 'triste', 'feliz', 'enojado']
        
        return {
            'empathy_training': sum(1 for w in empathy_words if w in words) / len(words) * 100,
            'knowledge_base': sum(1 for w in knowledge_words if w in words) / len(words) * 50,
            'dialogue_patterns': sum(1 for w in dialogue_words if w in words) / len(words) * 100,
            'emotional_context': sum(1 for w in emotion_words if w in words) / len(words) * 200
        }
    
    def _determine_training_priority(self, quality_score, category_scores, conversation_potential):
        """Determina prioridad con umbrales bajos"""
        max_category_score = max(category_scores.values()) if category_scores else 0
        avg_conversation_potential = np.mean(list(conversation_potential.values()))
        
        priority_score = (quality_score * 0.4 + max_category_score * 0.3 + avg_conversation_potential * 0.3)
        
        if priority_score >= 0.3:
            return 'high'
        elif priority_score >= 0.15:
            return 'medium'
        else:
            return 'low'
    
    def process_large_corpus(self, input_file, output_file, batch_size=1000):
        """Procesa corpus grande por streaming real - maneja JSON con indentación"""
        print(f"Procesando corpus grande: {input_file}")
        
        analyzed_articles = []
        stats = {
            'total_articles': 0,
            'valid_articles': 0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'category_distribution': defaultdict(int),
            'priority_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_words': 0
        }
        
        batch = []
        total_processed = 0
        
        # Variables para leer JSON con indentación
        current_article = []
        inside_article = False
        brace_count = 0
        
        print("Procesando JSON con indentación por streaming...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Saltar líneas vacías y el array inicial
                if not line or line == '[' or line == ']':
                    continue
                
                # Detectar inicio de artículo
                if line.startswith('{'):
                    inside_article = True
                    brace_count = 1
                    current_article = [line]
                    continue
                
                # Si estamos dentro de un artículo
                if inside_article:
                    current_article.append(line)
                    
                    # Contar llaves para saber cuándo termina el objeto
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                    
                    # Si las llaves se balancean, terminó el artículo
                    if brace_count == 0:
                        # Unir todas las líneas del artículo
                        article_json = ' '.join(current_article)
                        
                        # Limpiar coma final si existe
                        if article_json.endswith(','):
                            article_json = article_json[:-1]
                        
                        try:
                            # Parsear el artículo individual
                            article = json.loads(article_json)
                            batch.append(article)
                            
                            # Procesar batch cuando alcance el tamaño
                            if len(batch) >= batch_size:
                                self._process_batch(batch, analyzed_articles, stats)
                                total_processed += len(batch)
                                print(f"Procesados: {total_processed:,}, Válidos: {stats['valid_articles']:,}")
                                batch = []
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parseando artículo en línea {line_num}: {e}")
                        except Exception as e:
                            print(f"Error procesando artículo en línea {line_num}: {e}")
                        
                        # Reset para el siguiente artículo
                        inside_article = False
                        current_article = []
                        brace_count = 0
                
                # Mostrar progreso cada 10000 líneas
                if line_num % 10000 == 0:
                    print(f"Líneas leídas: {line_num:,}")
        
        # Procesar último batch
        if batch:
            self._process_batch(batch, analyzed_articles, stats)
            total_processed += len(batch)
        
        # Calcular estadísticas finales
        if stats['valid_articles'] > 0:
            stats['avg_quality'] = np.mean([a['analysis']['quality_score'] for a in analyzed_articles])
        else:
            stats['avg_quality'] = 0
        
        # Guardar resultados
        output_data = {
            'metadata': {
                'processed_date': datetime.now().isoformat(),
                'total_articles': stats['valid_articles'],
                'statistics': stats
            },
            'articles': analyzed_articles
        }
        
        print(f"Guardando {len(analyzed_articles)} artículos analizados...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAnálisis completado:")
        print(f"  Artículos válidos: {stats['valid_articles']:,}")
        print(f"  Calidad promedio: {stats['avg_quality']:.3f}")
        print(f"  Archivo guardado: {output_file}")
        
        return output_data
    
    def _process_batch(self, batch, analyzed_articles, stats):
        """Procesa un batch de artículos"""
        for article in batch:
            stats['total_articles'] += 1
            
            analysis = self.analyze_article(article)
            if analysis is None:
                continue
            
            # Artículo válido
            enriched_article = {**article, 'analysis': analysis}
            analyzed_articles.append(enriched_article)
            stats['valid_articles'] += 1
            
            # Actualizar estadísticas
            quality = analysis['quality_score']
            if quality >= 0.5:
                stats['quality_distribution']['high'] += 1
            elif quality >= 0.25:
                stats['quality_distribution']['medium'] += 1
            else:
                stats['quality_distribution']['low'] += 1
            
            stats['category_distribution'][analysis['primary_category']] += 1
            stats['priority_distribution'][analysis['training_priority']] += 1
            stats['total_words'] += analysis['metrics']['word_count']

def main():
    parser = argparse.ArgumentParser(description='Analizador streaming para corpus grandes')
    parser.add_argument('input_file', help='Archivo JSON grande de Wikipedia')
    parser.add_argument('-o', '--output', default='corpus_analyzed.json')
    parser.add_argument('--batch-size', type=int, default=1000, help='Tamaño de batch para procesamiento')
    
    args = parser.parse_args()
    
    analyzer = StreamingMIAAnalyzer()
    analyzer.process_large_corpus(args.input_file, args.output, args.batch_size)

if __name__ == "__main__":
    main()