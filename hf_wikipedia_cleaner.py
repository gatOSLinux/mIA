#!/usr/bin/env python3
"""
Limpiador de Wikipedia que puede usar dataset de Hugging Face o archivo local
"""

import bz2
import re
import json
import os
import html
import argparse
from datetime import datetime
from datasets import load_dataset
import requests

def clean_text(text):
    """Limpia el texto de wikitext completamente"""
    if not text:
        return ""
    
    # 1. Decodificar HTML
    text = html.unescape(text)
    
    # 2. Remover templates {{...}} y restos
    text = re.sub(r'\{\{(?:[^{}]|\{[^{}]*\})*\}\}', '', text)
    text = re.sub(r'\}\}', '', text)  # }} sueltos
    
    # 3. Remover referencias
    text = re.sub(r'<ref[^>]*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*?/>', '', text)
    
    # 4. Remover otros tags HTML
    text = re.sub(r'<[^>]*?>', '', text)
    
    # 5. Procesar enlaces [[...]]
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[link|texto]] -> texto
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)             # [[link]] -> link
    
    # 6. Remover enlaces externos
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)  # [url texto] -> texto
    text = re.sub(r'\[https?://[^\s\]]+\]', '', text)                # [url] -> 
    text = re.sub(r'https?://[^\s]+', '', text)                      # url solitaria
    
    # 7. Limpiar formato wiki
    text = re.sub(r"'{5}([^']+)'{5}", r'\1', text)    # '''''texto''''' -> texto
    text = re.sub(r"'{3}([^']+)'{3}", r'\1', text)    # '''texto''' -> texto
    text = re.sub(r"'{2}([^']+)'{2}", r'\1', text)    # ''texto'' -> texto
    
    # 8. Remover encabezados === ===
    text = re.sub(r'^=+\s*([^=]+)\s*=+\s*$', r'\1', text, flags=re.MULTILINE)
    
    # 9. Remover marcado de listas
    text = re.sub(r'^\s*[\*#:;]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\{\|\!\-\+\|].*$', '', text, flags=re.MULTILINE)
    
    # 10. Remover categorías, archivos e imágenes
    text = re.sub(r'\[\[Categoría:[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[Archivo:[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[Imagen:[^\]]+\]\]', '', text)
    text = re.sub(r'miniaturadeimagen\|[^|]*\|?', '', text)          # miniaturadeimagen|270px|
    text = re.sub(r'thumb\|[^|]*\|?', '', text)                     # thumb|270px|
    text = re.sub(r'^thumb\|.*$', '', text, flags=re.MULTILINE)     # thumb| al inicio
    text = re.sub(r'\d+px\|', '', text)                             # 300px| restos
    text = re.sub(r'Categoría:[^\n]*', '', text)                    # Categoría: al final
    
    # 11. Remover metadatos y códigos
    text = re.sub(r'^[A-Z\s]+::\s*[A-Z\s]+$', '', text, flags=re.MULTILINE)  # Europe :: ANDORRA
    text = re.sub(r'^[a-z]{2,3}:[^\s]+.*$', '', text, flags=re.MULTILINE)     # bn:অ্যান্ডোরা
    
    # 12. Remover referencias específicas
    text = re.sub(r'CIA\s*-\s*The World Factbook.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\([a-z\s]*inglés[a-z\s]*\)', '', text)
    text = re.sub(r'\([a-z\s]*español[a-z\s]*\)', '', text)
    text = re.sub(r'\([a-z\s]*catalán[a-z\s]*\)', '', text)
    
    # 13. Limpiar espacios
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 14. Filtrar líneas por contenido
    lines = []
    skip_sections = False
    
    for line in text.split('\n'):
        line = line.strip()
        
        # Saltar secciones de referencias
        line_lower = line.lower()
        section_triggers = [
            'vease tambien', 'véase también', 'referencias', 'bibliografia', 'bibliografía',
            'enlaces externos', 'notas', 'fuentes', 'external links', 'see also',
            'portal de', 'página del gobierno', 'categoría:'
        ]
        
        if any(trigger in line_lower for trigger in section_triggers):
            skip_sections = True
            continue
        
        if skip_sections:
            continue
        
        # Mantener líneas útiles
        if (len(line) > 15 and 
            not re.match(r'^[\s\W]*$', line) and
            not re.match(r'^[A-Z\s]+::\s*[A-Z\s]+$', line)):
            lines.append(line)
    
    return '\n'.join(lines).strip()

def is_valid_article(title, text):
    """Valida si el artículo es útil"""
    if not title or not text:
        return False
    
    # Filtrar namespaces
    if any(title.startswith(ns) for ns in [
        'Wikipedia:', 'Wikiproyecto:', 'Plantilla:', 'Categoría:', 'Archivo:', 
        'Usuario:', 'Discusión:', 'Anexo:', 'Portal:', 'Módulo:'
    ]):
        return False
    
    # Filtrar redirects
    if text.strip().lower().startswith(('#redirect', '#redirección')):
        return False
    
    # Filtrar desambiguación
    if 'desambiguación' in text.lower():
        return False
    
    # Filtrar muy cortos
    if len(text) < 200:
        return False
    
    return True

def download_from_huggingface(dataset_name, output_file):
    """Descarga el dataset desde Hugging Face"""
    print(f"Descargando dataset desde Hugging Face: {dataset_name}")
    
    try:
        # Descargar archivo directamente
        url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/eswiki-latest-pages-articles.xml.bz2"
        
        print(f"Descargando desde: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Guardar archivo
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Dataset descargado: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        return None

def process_wikipedia_from_hf(dataset_name, output_file, max_articles=None):
    """Procesa Wikipedia desde Hugging Face dataset"""
    
    # Descargar dataset si no existe localmente
    local_file = "eswiki-latest-pages-articles.xml.bz2"
    if not os.path.exists(local_file):
        downloaded = download_from_huggingface(dataset_name, local_file)
        if not downloaded:
            print("Error: No se pudo descargar el dataset")
            return 0
    else:
        print(f"Usando archivo local: {local_file}")
    
    # Procesar el archivo descargado
    return process_wikipedia_file(local_file, output_file, max_articles)

def process_wikipedia_file(input_file, output_file, max_articles=None):
    """Procesa un archivo local de Wikipedia"""
    print(f"Procesando {input_file} → {output_file}")
    
    processed = 0
    valid = 0
    
    with bz2.open(input_file, 'rt', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        f_out.write('[\n')
        first = True
        
        # Variables de parseo
        in_page = False
        current_title = ""
        current_text = ""
        
        for line in f_in:
            line = line.strip()
            
            if '<page>' in line:
                in_page = True
                current_title = ""
                current_text = ""
            
            elif '<title>' in line and '</title>' in line:
                match = re.search(r'<title>(.*?)</title>', line)
                if match:
                    current_title = match.group(1)
            
            elif '<text' in line and 'xml:space="preserve"' in line:
                if '</text>' in line:
                    match = re.search(r'xml:space="preserve"[^>]*>(.*?)</text>', line, re.DOTALL)
                    if match:
                        current_text = match.group(1)
                else:
                    start = line.find('>') + 1
                    current_text = line[start:]
                    for next_line in f_in:
                        if '</text>' in next_line:
                            current_text += '\n' + next_line.replace('</text>', '')
                            break
                        current_text += '\n' + next_line
            
            elif '</page>' in line and in_page:
                processed += 1
                
                if current_title and current_text:
                    if is_valid_article(current_title, current_text):
                        clean_content = clean_text(current_text)
                        
                        if len(clean_content) > 100:
                            article = {
                                "title": current_title,
                                "content": clean_content,
                                "word_count": len(clean_content.split()),
                                "char_count": len(clean_content),
                                "source": "wikipedia_es",
                                "processed_date": datetime.now().isoformat()[:10]
                            }
                            
                            if not first:
                                f_out.write(',\n')
                            else:
                                first = False
                            
                            json.dump(article, f_out, ensure_ascii=False, indent=2)
                            valid += 1
                            
                            if valid % 1000 == 0:
                                print(f"Válidos: {valid:,}")
                            
                            if max_articles and valid >= max_articles:
                                break
                
                if processed % 5000 == 0:
                    print(f"Procesados: {processed:,}")
                
                in_page = False
        
        f_out.write('\n]')
    
    print(f"\nCompletado: {valid:,} artículos válidos de {processed:,} procesados")
    return valid

def main():
    parser = argparse.ArgumentParser(description='Limpiar Wikipedia desde Hugging Face o archivo local')
    parser.add_argument('--hf-dataset', 
                       default='masterPokemonGlinux/wikipedia_articles',
                       help='Dataset de Hugging Face (default: masterPokemonGlinux/wikipedia_articles)')
    parser.add_argument('--local-file', 
                       help='Archivo local .xml.bz2 (si se especifica, no usa HF)')
    parser.add_argument('-o', '--output', 
                       default='wikipedia_clean.json',
                       help='Archivo JSON de salida')
    parser.add_argument('-m', '--max-articles', type=int,
                       help='Máximo número de artículos válidos')
    
    args = parser.parse_args()
    
    # Decidir fuente de datos
    if args.local_file:
        if not os.path.exists(args.local_file):
            print(f"Error: {args.local_file} no encontrado")
            return
        print(f"Procesando archivo local: {args.local_file}")
        process_wikipedia_file(args.local_file, args.output, args.max_articles)
    else:
        print(f"Procesando dataset de Hugging Face: {args.hf_dataset}")
        process_wikipedia_from_hf(args.hf_dataset, args.output, args.max_articles)

if __name__ == "__main__":
    main()