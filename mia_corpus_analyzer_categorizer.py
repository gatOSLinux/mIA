#!/usr/bin/env python3
"""
Analizador y categorizador de corpus para MIA
Analiza el JSON limpio y lo categoriza para optimizar el entrenamiento
"""

import json
import re
import os
from collections import Counter, defaultdict
import argparse
from datetime import datetime
import numpy as np

class MIACorpusAnalyzer:
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
        
        # Filtros de calidad para entrenamiento
        self.quality_thresholds = {
            'min_words': 50,        # Mínimo para ser útil
            'max_words': 2000,      # Máximo para mantener contexto
            'min_sentences': 3,     # Estructura mínima
            'diversity_threshold': 0.4  # Diversidad léxica mínima
        }
    
    def analyze_article(self, article):
        """Analiza un artículo y extrae métricas"""
        content = article['content']
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
        
        # Puntuación de calidad para entrenamiento
        quality_score = self._calculate_training_quality(metrics, content)
        
        # Categorización
        category_scores = self._categorize_content(content)
        primary_category = max(category_scores, key=category_scores.get) if category_scores else 'general'
        
        # Potencial conversacional específico para MIA
        conversation_potential = self._assess_mia_potential(content)
        
        return {
            'metrics': metrics,
            'quality_score': quality_score,
            'primary_category': primary_category,
            'category_scores': category_scores,
            'conversation_potential': conversation_potential,
            'training_priority': self._determine_training_priority(quality_score, category_scores, conversation_potential)
        }
    
    def _calculate_training_quality(self, metrics, content):
        """Calcula calidad específica para entrenamiento de LLM"""
        score = 0.0
        
        # Longitud óptima para entrenamiento (peso: 0.3)
        word_count = metrics['word_count']
        if 100 <= word_count <= 500:
            score += 0.3
        elif 50 <= word_count < 100 or 500 < word_count <= 1000:
            score += 0.2
        elif 1000 < word_count <= 2000:
            score += 0.1
        
        # Diversidad léxica (peso: 0.25)
        if metrics['lexical_diversity'] >= 0.6:
            score += 0.25
        elif metrics['lexical_diversity'] >= 0.4:
            score += 0.15
        elif metrics['lexical_diversity'] >= 0.3:
            score += 0.1
        
        # Estructura de oraciones (peso: 0.2)
        avg_length = metrics['avg_sentence_length']
        if 10 <= avg_length <= 20:
            score += 0.2
        elif 8 <= avg_length < 10 or 20 < avg_length <= 25:
            score += 0.15
        elif 5 <= avg_length < 8 or 25 < avg_length <= 30:
            score += 0.1
        
        # Contenido conversacional (peso: 0.15)
        conversational_markers = ['es', 'son', 'puede', 'debe', 'tiene', 'hace', 'dice', 'significa']
        marker_density = sum(1 for marker in conversational_markers if marker in content.lower()) / word_count
        score += 0.15 * min(marker_density * 100, 1.0)
        
        # Gramática y puntuación (peso: 0.1)
        if re.search(r'[.!?]', content) and re.search(r'[,;:]', content):
            score += 0.1
        elif re.search(r'[.!?]', content):
            score += 0.05
        
        return min(score, 1.0)
    
    def _categorize_content(self, content):
        """Categoriza el contenido según utilidad para MIA"""
        content_lower = content.lower()
        words = content_lower.split()
        word_set = set(words)
        
        category_scores = {}
        
        for category, info in self.categories.items():
            # Contar coincidencias de keywords
            matches = sum(1 for keyword in info['keywords'] if keyword in word_set)
            # Normalizar por longitud del contenido
            score = matches / len(words) * 1000 if words else 0
            category_scores[category] = score
        
        return category_scores
    
    def _assess_mia_potential(self, content):
        """Evalúa potencial específico para MIA"""
        content_lower = content.lower()
        words = content_lower.split()
        
        potential = {
            'empathy_training': 0.0,      # Para entrenar respuestas empáticas
            'knowledge_base': 0.0,        # Para base de conocimiento
            'dialogue_patterns': 0.0,     # Para patrones de diálogo
            'emotional_context': 0.0      # Para contexto emocional
        }
        
        # Potencial empático
        empathy_words = ['sentir', 'emoción', 'ayuda', 'apoyo', 'comprende', 'escucha', 'acompaña']
        potential['empathy_training'] = sum(1 for word in empathy_words if word in words) / len(words) * 100
        
        # Base de conocimiento
        knowledge_words = ['es', 'son', 'significa', 'consiste', 'permite', 'causa', 'efecto', 'caracteriza']
        potential['knowledge_base'] = sum(1 for word in knowledge_words if word in words) / len(words) * 50
        
        # Patrones de diálogo
        dialogue_words = ['dice', 'habla', 'conversa', 'pregunta', 'responde', 'explica', 'cuenta']
        potential['dialogue_patterns'] = sum(1 for word in dialogue_words if word in words) / len(words) * 100
        
        # Contexto emocional
        emotion_words = ['alegre', 'triste', 'feliz', 'enojado', 'ansioso', 'tranquilo', 'emocionado']
        potential['emotional_context'] = sum(1 for word in emotion_words if word in words) / len(words) * 200
        
        return potential
    
    def _determine_training_priority(self, quality_score, category_scores, conversation_potential):
        """Determina prioridad para entrenamiento"""
        # Combinar métricas para prioridad
        max_category_score = max(category_scores.values()) if category_scores else 0
        avg_conversation_potential = np.mean(list(conversation_potential.values()))
        
        priority_score = (quality_score * 0.4 + 
                         max_category_score * 0.3 + 
                         avg_conversation_potential * 0.3)
        
        if priority_score >= 0.7:
            return 'high'
        elif priority_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def process_corpus(self, input_file, output_file):
        """Procesa todo el corpus y genera análisis completo"""
        print(f"Analizando corpus: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Artículos cargados: {len(articles)}")
        
        analyzed_articles = []
        stats = {
            'total_articles': len(articles),
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'category_distribution': defaultdict(int),
            'priority_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_words': 0,
            'avg_quality': 0
        }
        
        for i, article in enumerate(articles):
            # Aplicar filtros de calidad básicos
            if not self._passes_basic_filters(article):
                continue
            
            # Analizar artículo
            analysis = self.analyze_article(article)
            
            # Crear artículo enriquecido
            enriched_article = {
                **article,
                'analysis': analysis
            }
            
            analyzed_articles.append(enriched_article)
            
            # Actualizar estadísticas
            quality = analysis['quality_score']
            if quality >= 0.7:
                stats['quality_distribution']['high'] += 1
            elif quality >= 0.4:
                stats['quality_distribution']['medium'] += 1
            else:
                stats['quality_distribution']['low'] += 1
            
            stats['category_distribution'][analysis['primary_category']] += 1
            stats['priority_distribution'][analysis['training_priority']] += 1
            stats['total_words'] += analysis['metrics']['word_count']
            
            if (i + 1) % 1000 == 0:
                print(f"Analizados: {i + 1:,}")
        
        stats['avg_quality'] = np.mean([a['analysis']['quality_score'] for a in analyzed_articles])
        
        # Guardar corpus analizado
        output_data = {
            'metadata': {
                'processed_date': datetime.now().isoformat(),
                'total_articles': len(analyzed_articles),
                'statistics': stats
            },
            'articles': analyzed_articles
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAnálisis completado:")
        print(f"  Artículos analizados: {len(analyzed_articles):,}")
        print(f"  Calidad promedio: {stats['avg_quality']:.3f}")
        print(f"  Alta prioridad: {stats['priority_distribution']['high']:,}")
        print(f"  Archivo guardado: {output_file}")
        
        return output_data
    
    def _passes_basic_filters(self, article):
        """Filtros básicos de calidad"""
        word_count = article.get('word_count', 0)
        content = article.get('content', '')
        
        if word_count < self.quality_thresholds['min_words']:
            return False
        if word_count > self.quality_thresholds['max_words']:
            return False
        
        # Filtrar contenido muy repetitivo
        words = content.split()
        if len(set(words)) / len(words) < 0.3:
            return False
        
        return True
    
    def create_training_subsets(self, analyzed_corpus_file, output_dir):
        """Crea subconjuntos optimizados para entrenamiento"""
        with open(analyzed_corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data['articles']
        os.makedirs(output_dir, exist_ok=True)
        
        # Subset 1: Alta calidad para pre-entrenamiento
        high_quality = [a for a in articles if a['analysis']['quality_score'] >= 0.6]
        with open(f"{output_dir}/high_quality_pretrain.json", 'w', encoding='utf-8') as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        
        # Subset 2: Contenido conversacional para fine-tuning
        conversational = [a for a in articles if a['analysis']['primary_category'] in ['conversational', 'emotional', 'social']]
        with open(f"{output_dir}/conversational_finetune.json", 'w', encoding='utf-8') as f:
            json.dump(conversational, f, ensure_ascii=False, indent=2)
        
        # Subset 3: Contenido educativo para base de conocimiento
        educational = [a for a in articles if a['analysis']['primary_category'] in ['educational', 'cultural', 'biographical']]
        with open(f"{output_dir}/educational_knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(educational, f, ensure_ascii=False, indent=2)
        
        print(f"Subconjuntos creados en {output_dir}:")
        print(f"  Alta calidad: {len(high_quality):,} artículos")
        print(f"  Conversacional: {len(conversational):,} artículos") 
        print(f"  Educativo: {len(educational):,} artículos")

def main():
    parser = argparse.ArgumentParser(description='Analizar y categorizar corpus para MIA')
    parser.add_argument('input_file', help='Archivo JSON limpio de Wikipedia')
    parser.add_argument('-o', '--output', default='corpus_analyzed.json',
                       help='Archivo de salida con análisis')
    parser.add_argument('--create-subsets', action='store_true',
                       help='Crear subconjuntos para entrenamiento')
    parser.add_argument('--subset-dir', default='./training_subsets',
                       help='Directorio para subconjuntos')
    
    args = parser.parse_args()
    
    analyzer = MIACorpusAnalyzer()
    
    # Analizar corpus completo
    output_data = analyzer.process_corpus(args.input_file, args.output)
    
    # Crear subconjuntos si se solicita
    if args.create_subsets:
        analyzer.create_training_subsets(args.output, args.subset_dir)

if __name__ == "__main__":
    main()