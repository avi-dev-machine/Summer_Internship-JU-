'''Whisper Offline Transcriber
A simple GUI application to record audio and transcribe it using Whisper.
Author: Avijnan Purkait'''

import tkinter as tk
from tkinter import messagebox, scrolledtext
import whisper
import sounddevice as sd
import numpy as np
import wave
import tempfile
import threading
import time
import os

RECORD_SECONDS = 5
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_NAME = "base"  # You can change to "tiny" for faster results

model = whisper.load_model(MODEL_NAME)

def record_audio(filename, duration=RECORD_SECONDS, fs=SAMPLE_RATE):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype='int16')
    sd.wait()

    # Save WAV file
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        print(f"✅ Audio saved at {filename}")
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        raise

def transcribe_audio(filepath):
    print("🧠 Transcribing:", filepath)

    # Ensure file exists and is accessible
    retries = 5
    for _ in range(retries):
        if os.path.exists(filepath):
            try:
                return model.transcribe(filepath)["text"]
            except Exception as e:
                print(f"Whisper load retry due to: {e}")
                time.sleep(1)
        else:
            time.sleep(0.5)

    raise FileNotFoundError(f"Audio file not found: {filepath}")

def start_transcription():
    btn_record.config(state=tk.DISABLED)
    output_box.delete('1.0', tk.END)
    output_box.insert(tk.END, "🎤 Listening...\n")

    def task():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                audio_path = tf.name

            record_audio(audio_path)
            text = transcribe_audio(audio_path)
            output_box.delete('1.0', tk.END)
            output_box.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))
        finally:
            btn_record.config(state=tk.NORMAL)

    threading.Thread(target=task).start()

def on_shortcut(event=None):
    start_transcription()

# GUI Setup
root = tk.Tk()
root.title("🎤 Whisper Offline Transcriber")
root.geometry("520x350")
root.resizable(False, False)

tk.Label(root, text="Whisper Offline Transcriber", font=("Segoe UI", 16, "bold")).pack(pady=10)

btn_record = tk.Button(root, text="🎙️ Start Listening (Ctrl+M)", font=("Segoe UI", 12), command=start_transcription)
btn_record.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=12, font=("Consolas", 11))
output_box.pack(padx=10, pady=10)

root.bind("<Control-m>", on_shortcut)
root.mainloop()
