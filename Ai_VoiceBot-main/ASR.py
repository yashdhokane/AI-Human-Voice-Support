# asr/asr.py

import vosk
import json
import numpy as np


class ASR:
    def __init__(self, model_path="D:\\vosk-model-small-en-us-0.15", sample_rate=16000):
        """
        ASR class using Vosk.
        """
        self.sample_rate = sample_rate
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

        # Optional: improve final sentence quality
        self.recognizer.SetWords(True)

    def process(self, audio_chunk: bytes) -> str:
        """
        Processes a full buffered sentence and returns clean transcription.
        Normalizes audio to improve recognition.
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

        # Normalize volume
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Convert back to int16 bytes for Vosk
        audio = (audio * 32767).astype(np.int16).tobytes()

        # Feed to recognizer
        self.recognizer.AcceptWaveform(audio)

        # Get final text
        result = json.loads(self.recognizer.Result())
        text = result.get("text", "").strip()

        # Reset recognizer for next sentence
        self.recognizer.Reset()

        return text
