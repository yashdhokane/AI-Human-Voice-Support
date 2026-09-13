import os
import tempfile
import sounddevice as sd
import soundfile as sf
import asyncio
import edge_tts

VOICE = "en-US-JennyNeural"
SPEAKER_DEVICE = 4  # Speakers (Realtek(R) Audio)


def speak(text: str):
    """Speak text aloud. Blocks until done."""
    print(f"[TTS] 🔊 {text[:80]}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name

    try:
        # Run edge-tts in a fresh event loop in current thread
        async def _run():
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(tmp)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()

        data, sr = sf.read(tmp)
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        sd.play(data, sr, device=SPEAKER_DEVICE)
        sd.wait()
    except Exception as e:
        print(f"[TTS] ❌ {e}")
        # Fallback: try pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e2:
            print(f"[TTS] ❌ Fallback also failed: {e2}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    speak("Hello! I am your T-Mobile billing assistant. How can I help you today?")