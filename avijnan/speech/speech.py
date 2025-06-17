'''
Real-time speech-to-text transcription using OpenAI Whisper.
Records audio from microphone, transcribes to text, and cleans up temporary files.
Provides both class-based and simple function interfaces for speech input.

Author:AvijnanPurkait
'''
import whisper
import sounddevice as sd
import wave
import tempfile
import threading
import time
import os

class SpeechTranscriber:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.sample_rate = 16000
        self.channels = 1
        self.model = whisper.load_model(model_name)
    
    def record_audio(self, duration, filename=None):
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            filename = temp_file.name
            temp_file.close()
        
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=self.channels, dtype='int16')
        sd.wait()
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        
        return filename
    
    def transcribe_file(self, filepath):
        for _ in range(5):
            if os.path.exists(filepath):
                try:
                    result = self.model.transcribe(filepath)
                    return result["text"].strip()
                except Exception:
                    time.sleep(1)
            else:
                time.sleep(0.5)
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    def listen_and_transcribe(self, duration=5.0):
        audio_file = None
        try:
            audio_file = self.record_audio(duration)
            text = self.transcribe_file(audio_file)
            return text
        finally:
            if audio_file and os.path.exists(audio_file):
                try:
                    os.unlink(audio_file)
                except:
                    pass

# Global instance for simple usage
_transcriber = None

def get_speech_input(duration=5.0, model="base"):
    global _transcriber
    if _transcriber is None or _transcriber.model_name != model:
        _transcriber = SpeechTranscriber(model)
    return _transcriber.listen_and_transcribe(duration)