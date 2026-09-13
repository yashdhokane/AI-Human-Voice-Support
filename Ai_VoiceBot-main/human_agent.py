import asyncio
import json
import queue
import threading
import os
import sys
import sounddevice as sd
import vosk
from dotenv import load_dotenv
from livekit import rtc

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
VOSK_MODEL_PATH = r"D:\vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
MIC_DEVICE = 1

print("Loading Vosk model...")
model = vosk.Model(VOSK_MODEL_PATH)
print("Vosk model loaded! ✅")

audio_q = queue.Queue()
stop_event = threading.Event()


def audio_callback(indata, frames, time, status):
    if not stop_event.is_set():
        try:
            audio_q.put(bytes(indata))
        except Exception:
            pass


async def run_human_agent(token: str):
    room = rtc.Room()
    loop = asyncio.get_running_loop()
    disconnected = asyncio.Event()

    print("\n" + "="*50)
    print("🎧 HUMAN AGENT CONSOLE")
    print("="*50)
    print("🎙️ Speak into microphone to talk to user")
    print("⌨️  Type 'handover' to transfer back to AI bot")
    print("⌨️  Type 'exit' to end call")
    print("="*50 + "\n")

    @room.on("data_received")
    def on_data(data: rtc.DataPacket, **kwargs):
        message = data.data.decode("utf-8")
        print(f"\n📩 {message}\n")

    @room.on("disconnected")
    def on_disconnected(**kwargs):
        loop.call_soon_threadsafe(disconnected.set)

    await room.connect(LIVEKIT_URL, token)
    print("✅ Connected to room!\n")

    def listen_and_send():
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=4000,
            device=MIC_DEVICE,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ):
            print("🎙️ Agent mic active - speak now!")
            while not stop_event.is_set():
                try:
                    data = audio_q.get(timeout=1)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            print(f"[Agent said]: {text}")
                            asyncio.run_coroutine_threadsafe(
                                room.local_participant.publish_data(
                                    f"Agent: {text}".encode("utf-8")
                                ),
                                loop
                            )
                except queue.Empty:
                    continue
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"Audio error: {e}")
                    break

    voice_thread = threading.Thread(target=listen_and_send, daemon=True)
    voice_thread.start()

    # Handle commands
    # Handle commands
    while not disconnected.is_set():
        try:
            cmd = await loop.run_in_executor(None, input, "CMD> ")
            cmd = cmd.strip().lower()
            print(f"[DEBUG] Command received: '{cmd}'")

            if cmd == "handover":
                print("\n🔄 Transferring back to AI Bot...")
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        room.local_participant.publish_data(
                            "HANDOVER_TO_BOT".encode("utf-8")
                        ),
                        loop
                    )
                    future.result(timeout=5)
                    print("[DEBUG] Handover signal sent!")
                except Exception as e:
                    print(f"[DEBUG] Publish error: {e}")
                    # Try direct await as fallback
                    await room.local_participant.publish_data(
                        "HANDOVER_TO_BOT".encode("utf-8")
                    )
                await asyncio.sleep(2)
                stop_event.set()
                await room.disconnect()
                print("✅ Handover complete!")
                break

            elif cmd == "exit":
                print("\n❌ Ending call...")
                stop_event.set()
                await room.disconnect()
                break

            else:
                print(f"Unknown command: '{cmd}'. Type 'handover' or 'exit'")

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[DEBUG] Command error: {e}")
            break

    stop_event.set()
    print("Agent session ended.")

    stop_event.set()
    print("Agent session ended.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python human_agent.py <token>")
        sys.exit(1)
    try:
        asyncio.run(run_human_agent(sys.argv[1]))
    except KeyboardInterrupt:
        print("\nAgent disconnected.")