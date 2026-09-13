import json
import queue
import sounddevice as sd
import vosk
import threading

VOSK_MODEL_PATH = r"D:\vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
MIC_DEVICE = 1      # Realtek Audio Mic
SPEAKER_DEVICE = 4    # Realtek Speakers

# Load model once
print("Loading Vosk speech model...")
model = vosk.Model(VOSK_MODEL_PATH)
print("Vosk model loaded! ✅")


class VoiceListener:
    def __init__(self, on_speech_callback, label="User"):
        self.on_speech = on_speech_callback
        self.label = label
        self.running = False
        self._thread = None
        self._q = queue.Queue()

    def _audio_callback(self, indata, frames, time, status):
        self._q.put(bytes(indata))

    def _listen_loop(self):
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=4000,
            device=MIC_DEVICE,
            dtype="int16",
            channels=1,
            callback=self._audio_callback
        ):
            print(f"[{self.label}] 🎙️ Listening...")
            while self.running:
                data = self._q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"[{self.label}] Said: {text}")
                        self.on_speech(text)
                else:
                    partial = json.loads(rec.PartialResult())
                    p = partial.get("partial", "").strip()
                    if p:
                        print(f"[{self.label}] Hearing: {p}")

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        print(f"[{self.label}] 🔇 Stopped listening.")