"""
ai_bot.py — T-Mobile AI Voice Bot

Flow:
  1. Customer speaks → VAD detects → Vosk transcribes → RAG/Ollama answers → TTS speaks
  2. If AI can't handle → transfers to human agent (PC2)
  3. Human agent takes over, AI listens silently and logs context
  4. Customer says bye/end → call ends
  5. If human hands back → AI resumes with full context
"""

import asyncio
import json
import os
import queue
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import vosk
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from vad.VAD import MicrophoneVAD
from handover import ConversationHistory
from retrieval import answer_billing_question
from tts import speak

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
LIVEKIT_URL        = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "supersecretkey1234567890abcdef12")
PC2_IP             = os.getenv("PC2_IP", "192.168.1.86")
ROOM_NAME          = "billing-support-room"
VOSK_MODEL_PATH    = r"D:\vosk-model-small-en-us-0.15"
SAMPLE_RATE        = 16000
BLOCK_SIZE         = 3200  # 200ms blocks — more audio per chunk = better accuracy
MIC_DEVICE         = 1     # Microphone (Realtek USB Audio)
SPEAKER_DEVICE     = 4     # Speakers (Realtek(R) Audio)

END_CALL_PHRASES = [
    "goodbye", "bye", "end call", "hang up", "that's all",
    "thank you goodbye", "thanks goodbye", "no more questions"
]

# ─── Load Vosk ────────────────────────────────────────────────────────────────
print("Loading Vosk model...")
vosk_model = vosk.Model(VOSK_MODEL_PATH)
vosk.SetLogLevel(-1)  # suppress Vosk logs
print("✅ Vosk loaded!")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def generate_token(identity: str) -> str:
    grant = VideoGrants(room_join=True, room=ROOM_NAME)
    return (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(grant)
        .to_jwt()
    )


def notify_human_agent(agent_token: str):
    try:
        r = requests.post(f"http://{PC2_IP}:5000/start-agent",
                          json={"token": agent_token}, timeout=5)
        if r.status_code == 200:
            print("✅ Human agent notified!")
            return
    except Exception as e:
        print(f"❌ Can't reach PC2: {e}")
    print(f"📋 Run manually: python human_agent.py {agent_token}")


def is_end_call(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in END_CALL_PHRASES)


# ─── AI Bot ───────────────────────────────────────────────────────────────────
async def run_ai_bot():
    history     = ConversationHistory()
    vad         = MicrophoneVAD(silence_threshold=150, silence_duration=1.0)
    is_speaking = threading.Event()

    def speak_safe(text: str):
        """Run TTS in a fully isolated thread — no asyncio context."""
        done = threading.Event()

        def _run():
            # This thread has no event loop — safe for asyncio.new_event_loop()
            speak(text)
            time.sleep(0.5)
            done.set()

        is_speaking.set()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        done.wait()  # block caller until speech done
        is_speaking.clear()

    # ── Phase 1: AI Voice Bot ──────────────────────────────────────────────────
    async def ai_phase(room: rtc.Room, loop: asyncio.AbstractEventLoop) -> str:
        """
        AI handles the conversation.
        Returns: 'handover' | 'end_call'
        """
        result_event = asyncio.Event()
        result       = {"action": None}
        audio_q      = queue.Queue()
        stop         = threading.Event()
        buffer       = b""

        def audio_cb(indata, frames, time_info, status):
            if not stop.is_set():
                audio_q.put(bytes(indata))

        async def handle(text: str):
            if is_end_call(text):
                speak_safe("Thank you for calling T-Mobile. Goodbye!")
                result["action"] = "end_call"
                result_event.set()
                return

            history.add("user", text)
            ans = answer_billing_question(text, history)

            if ans["type"] == "handover_check":
                speak_safe("Let me connect you to a human agent. Please hold.")
                result["action"] = "handover"
                result_event.set()
            else:
                msg = ans["message"]
                history.add("assistant", msg)
                print(f"\n🤖 Bot: {msg}")
                threading.Thread(target=speak_safe, args=(msg,), daemon=True).start()

        def listen_loop():
            rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                                   device=MIC_DEVICE, dtype="int16",
                                   channels=1, callback=audio_cb):
                print("🎙️ Listening (AI phase)...")
                while not stop.is_set():
                    try:
                        data = audio_q.get(timeout=1)
                    except queue.Empty:
                        continue

                    if is_speaking.is_set():
                        continue

                    if rec.AcceptWaveform(data):
                        r = json.loads(rec.Result())
                        text = r.get("text", "").strip()
                        if text and len(text) > 2:
                            print(f"\n🗣 Customer: {text}")
                            asyncio.run_coroutine_threadsafe(handle(text), loop)
                    else:
                        partial = json.loads(rec.PartialResult())
                        p = partial.get("partial", "").strip()
                        if p:
                            print(f"  Hearing: {p}", end="\r")

        t = threading.Thread(target=listen_loop, daemon=True)
        t.start()

        await result_event.wait()
        stop.set()
        return result["action"]

    # ── Phase 2: Human Agent ───────────────────────────────────────────────────
    async def human_phase(room: rtc.Room, loop: asyncio.AbstractEventLoop) -> str:
        """
        Human agent handles the call.
        AI listens silently and logs context.
        Returns: 'resume' | 'end_call'
        """
        handover_back = asyncio.Event()
        end_call      = asyncio.Event()
        audio_q       = queue.Queue()
        stop          = threading.Event()
        buffer        = b""

        @room.on("data_received")
        def on_data(data: rtc.DataPacket, **kwargs):
            msg = data.data.decode("utf-8")
            if msg == "HANDOVER_TO_BOT":
                loop.call_soon_threadsafe(handover_back.set)
            elif msg.startswith("Agent:"):
                text = msg[len("Agent:"):].strip()
                if text:
                    history.add("agent", text)
                print(f"\n📩 {msg}")

        def audio_cb(indata, frames, time_info, status):
            if not stop.is_set():
                audio_q.put(bytes(indata))

        # AI silently transcribes customer during human phase
        def silent_listen():
            rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                                   device=MIC_DEVICE, dtype="int16",
                                   channels=1, callback=audio_cb):
                print("👂 AI listening silently (human phase)...")
                while not stop.is_set():
                    try:
                        data = audio_q.get(timeout=1)
                    except queue.Empty:
                        continue

                    if rec.AcceptWaveform(data):
                        r = json.loads(rec.Result())
                        text = r.get("text", "").strip()
                        if text and len(text) > 2:
                            print(f"\n[Logged] Customer: {text}")
                            history.add("user", text)
                            if is_end_call(text):
                                loop.call_soon_threadsafe(end_call.set)

        t = threading.Thread(target=silent_listen, daemon=True)
        t.start()

        # Wait for handover back or end call
        done, _ = await asyncio.wait(
            [
                asyncio.create_task(handover_back.wait()),
                asyncio.create_task(end_call.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop.set()

        if end_call.is_set():
            return "end_call"
        return "resume"

    # ── Main loop ──────────────────────────────────────────────────────────────
    room = rtc.Room()
    loop = asyncio.get_running_loop()

    disconnected = asyncio.Event()

    @room.on("disconnected")
    def on_disc(**kwargs):
        loop.call_soon_threadsafe(disconnected.set)

    # ── Connect to LiveKit ─────────────────────────────────────────────────────
    token = generate_token("ai-bot")
    await room.connect(LIVEKIT_URL, token)
    print(f"\n✅ Connected to room: {ROOM_NAME}\n")

    # ── Publish mic so human agent can hear customer ───────────────────────────
    audio_source = rtc.AudioSource(SAMPLE_RATE, 1)
    mic_track    = rtc.LocalAudioTrack.create_audio_track("customer", audio_source)
    await room.local_participant.publish_track(
        mic_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    # ── Continuously stream mic to LiveKit (always on, entire call) ───────────
    mic_stream_q   = queue.Queue()
    stop_mic_stream = threading.Event()

    def mic_stream_cb(indata, frames, time_info, status):
        if not stop_mic_stream.is_set():
            try:
                mic_stream_q.put_nowait(bytes(indata))
            except Exception:
                pass

    def mic_stream_loop():
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               device=MIC_DEVICE, dtype="int16",
                               channels=1, callback=mic_stream_cb):
            print("📡 Mic streaming to LiveKit...")
            while not stop_mic_stream.is_set():
                try:
                    data = mic_stream_q.get(timeout=1)
                except queue.Empty:
                    continue
                if is_speaking.is_set():
                    continue
                if loop.is_closed():
                    break
                pcm = np.frombuffer(data, dtype=np.int16)
                frame = rtc.AudioFrame(
                    data=pcm.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(pcm),
                )
                try:
                    asyncio.run_coroutine_threadsafe(
                        audio_source.capture_frame(frame), loop
                    )
                except Exception:
                    break

    threading.Thread(target=mic_stream_loop, daemon=True).start()

    # ── Play agent audio on customer's speaker ────────────────────────────────
    playback_q = queue.Queue()

    def playback_loop():
        current_sr = SAMPLE_RATE
        out_stream = None
        try:
            while not disconnected.is_set():
                try:
                    pcm, sr = playback_q.get(timeout=1)
                except queue.Empty:
                    continue
                # Reopen stream if sample rate changed
                if out_stream is None or sr != current_sr:
                    if out_stream:
                        out_stream.close()
                    current_sr = sr
                    try:
                        out_stream = sd.OutputStream(
                            samplerate=current_sr, channels=2,
                            dtype='float32', device=SPEAKER_DEVICE,
                            blocksize=0)
                        out_stream.start()
                        print(f"🔊 Speaker stream open at {current_sr}Hz")
                    except Exception as e:
                        print(f"Speaker error: {e}")
                        break
                try:
                    stereo = np.column_stack([pcm, pcm])
                    out_stream.write(stereo)
                except Exception as e:
                    print(f"Playback error: {e}")
                    break
        finally:
            if out_stream:
                out_stream.close()

    threading.Thread(target=playback_loop, daemon=True).start()

    @room.on("track_subscribed")
    def on_track(track, pub, participant, **kwargs):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"🔊 Audio from: {participant.identity}")
            asyncio.run_coroutine_threadsafe(_play(track), loop)

    async def _play(track):
        stream = rtc.AudioStream(track)
        async for ev in stream:
            if disconnected.is_set():
                break
            frame = ev.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                playback_q.put_nowait((pcm, frame.sample_rate))
            except Exception:
                pass

    # ── Greet ─────────────────────────────────────────────────────────────────
    threading.Thread(
        target=speak_safe,
        args=("Hello! Welcome to T-Mobile billing support. How can I help you today?",),
        daemon=True
    ).start()

    # ── State machine ──────────────────────────────────────────────────────────
    while not disconnected.is_set():
        # AI phase
        action = await ai_phase(room, loop)

        if action == "end_call":
            print("\n📵 Customer ended the call.")
            break

        if action == "handover":
            print("\n🔄 Transferring to human agent...")
            agent_token = generate_token("human-agent")
            notify_human_agent(agent_token)

            # Human phase
            action = await human_phase(room, loop)

            if action == "end_call":
                print("\n📵 Call ended during human phase.")
                speak_safe("Thank you for calling T-Mobile. Goodbye!")
                break

            if action == "resume":
                # Resume AI with context
                last = next(
                    (t["content"] for t in reversed(history.turns) if t["role"] == "user"),
                    None
                )
                msg = (
                    f"Welcome back! I've been listening. I understand you were discussing {last}. How can I help further?"
                    if last else
                    "Welcome back! How can I continue helping you?"
                )
                speak_safe(msg)
                # Continue outer while loop → ai_phase again

    stop_mic_stream.set()
    await room.disconnect()
    print("✅ Call ended. Goodbye!")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 T-Mobile AI Voice Bot")
    print("=" * 50)
    try:
        asyncio.run(run_ai_bot())
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")