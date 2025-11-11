import voice_recorder 
import speech_to_text
import inference_emotion_classifier 
import inference_agent_emotion_classifier
import json
#Pipeline
#Flujo PEAS
#Proceso
#Entrada
#Procesamiento
#Salida
def build_json_data(transcription_audio):
    data = {}
def main():
    print("Iniciando el programa...")
    voice_recorder.main()
    print("Procesando audio y transcribiendo...")
    text = speech_to_text.get_audio_transcription()

    print("Texto transcribido (VARIABLE):", text)
    print("Utilizando la primera red neuronal - Analisis de Sentimiento del Usuario")
    user_emotion = inference_emotion_classifier.predict(text)
    print("Emocion del usuario (VARIABLE):", user_emotion)
    user_agent = inference_agent_emotion_classifier.predict(text,user_emotion)
    print("Emocion del agente (VARIABLE):", user_agent)
    print("Finalizo el programa...")


if __name__ == "__main__":
    main()