import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

AUDIO_DIR = "files/audio"

def get_latest_audio_file():
    """Busca el archivo .mp3 más reciente en la carpeta files/audio."""
    mp3_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    if not mp3_files:
        raise FileNotFoundError("No se encontró ningún archivo .mp3 en files/audio")
    # Ordena por fecha de modificación (más reciente primero)
    mp3_files.sort(key=lambda f: os.path.getmtime(os.path.join(AUDIO_DIR, f)), reverse=True)
    return os.path.join(AUDIO_DIR, mp3_files[0])

def speech_to_text(audio_file_path):
    """Convierte un archivo de audio en texto usando Whisper."""
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="json",
            language="es"
        )
    return response.text

def get_audio_transcription():
    try:
        latest_audio = get_latest_audio_file()
        print(f"Transcribiendo: {latest_audio}")
        text = speech_to_text(latest_audio)
        print("\nTranscripción:\n", text)
        return text
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    get_audio_transcription()