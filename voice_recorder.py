# voice_recorder.py
# Graba audio al presionar 'F' (toggle) directamente a MP3 usando ffmpeg por stdin.
# Guarda en files/audio. 'F' para iniciar/detener y salir automáticamente al detener.
# 'Q' sigue saliendo manualmente si lo prefieres.
# No requiere sudo (usa pynput en vez de keyboard).

import os
import sys
import shutil
import queue
import subprocess
from datetime import datetime

import numpy as np
import sounddevice as sd
from pynput import keyboard

AUDIO_DIR = "files/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Configuración de audio
SAMPLE_RATE = 44100   # Hz
CHANNELS = 2
DTYPE = "int16"       # s16le para ffmpeg

# MP3 (LAME)
BITRATE = "192k"      # puedes usar 128k, 160k, 192k, 256k, etc.

class MP3Recorder:
    def __init__(self):
        self.recording = False
        self._q = queue.Queue()
        self._stream = None
        self._ffmpeg = None
        self.last_output = None

    def _ffmpeg_cmd(self, out_path: str):
        # ffmpeg leerá PCM crudo (s16le) desde stdin y escribirá MP3 (libmp3lame)
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", "pipe:0",
            "-c:a", "libmp3lame",
            "-b:a", BITRATE,
            "-y", out_path
        ]

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            # Asegurar int16
            if indata.dtype != np.int16:
                data = (indata * np.iinfo(np.int16).max).astype(np.int16)
            else:
                data = indata
            self._q.put(data.tobytes())

    def start(self):
        if self.recording:
            return
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg no está instalado o no está en PATH.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_mp3 = os.path.join(AUDIO_DIR, f"audio_{ts}.mp3")

        # Arranca ffmpeg
        self._ffmpeg = subprocess.Popen(
            self._ffmpeg_cmd(out_mp3),
            stdin=subprocess.PIPE
        )

        # Arranca stream de micro
        self._q.queue.clear()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback
        )
        self._stream.start()

        self.recording = True
        self.last_output = out_mp3
        print(f"🎙️ Grabando → {out_mp3}  (F para detener y salir)")

    def pump(self):
        """Bombear datos del queue hacia ffmpeg. Llamar regularmente en el loop principal."""
        if not self.recording:
            return
        try:
            while True:
                chunk = self._q.get_nowait()
                if self._ffmpeg and self._ffmpeg.stdin:
                    self._ffmpeg.stdin.write(chunk)
        except queue.Empty:
            pass

    def stop(self):
        if not self.recording:
            return
        self.recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._ffmpeg:
            try:
                if self._ffmpeg.stdin:
                    self._ffmpeg.stdin.close()
            except Exception:
                pass
            self._ffmpeg.wait()
            self._ffmpeg = None

        print(f"✅ Grabación finalizada: {self.last_output}")

    def delete_last(self):
        if self.last_output and os.path.exists(self.last_output):
            os.remove(self.last_output)
            print(f"🗑️ Eliminado: {self.last_output}")
            self.last_output = None
        else:
            print("ℹ️ No hay archivo previo que eliminar.")

def main():
    rec = MP3Recorder()
    print("Controles: F = iniciar/detener (y salir) | Q = salir")

    # Flags anti-rebote por autorepetición
    pressed = {"f": False, "q": False}

    # Usaremos una referencia al listener para poder pararlo desde on_press
    listener_ref = {"obj": None}

    def on_press(key):
        try:
            ch = key.char.lower()
        except AttributeError:
            return  # omitimos teclas especiales

        # Toggle grabación con F
        if ch == 'f' and not pressed["f"]:
            pressed["f"] = True
            if not rec.recording:
                rec.start()
            else:
                # Si estaba grabando y se presiona F, se detiene y SALIMOS
                rec.stop()
                print("👋 Saliendo…")
                # Detener el listener para terminar el loop principal
                if listener_ref["obj"]:
                    listener_ref["obj"].stop()
                return False  # asegura la salida

        # Salida manual con Q (por si el usuario prefiere salir sin grabar)
        elif ch == 'q' and not pressed["q"]:
            pressed["q"] = True
            if rec.recording:
                rec.stop()
            print("👋 Saliendo…")
            if listener_ref["obj"]:
                listener_ref["obj"].stop()
            return False

    def on_release(key):
        try:
            ch = key.char.lower()
        except AttributeError:
            return
        if ch in pressed:
            pressed[ch] = False

    # Listener de teclado y loop de bombeo de audio
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener_ref["obj"] = listener
        while listener.running:
            rec.pump()
            sd.sleep(20)  # evita alto uso de CPU

if __name__ == "__main__":
    main()

