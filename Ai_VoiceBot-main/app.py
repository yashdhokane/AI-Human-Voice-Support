import asyncio
import os
import subprocess
import time
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants
from retrieval import answer_billing_question
from handover import ConversationHistory

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://172.17.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "supersecretkey1234567890abcdef12")


def start_livekit():
    """Auto-start LiveKit server if not already running"""
    result = subprocess.run(
        ['netstat', '-an'],
        capture_output=True,
        text=True
    )
    if '7880' in result.stdout:
        print("✅ LiveKit already running!")
        return None

    print("🚀 Starting LiveKit server...")
    process = subprocess.Popen(
        [r'D:\livekit\livekit-server.exe', '--config', r'D:\livekit\livekit.yaml'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

    # Verify it started
    result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
    if '7880' in result.stdout:
        print("✅ LiveKit server started!")
    else:
        print("❌ LiveKit failed to start!")
        print(r"Please run manually: D:\livekit\livekit-server.exe --config D:\livekit\livekit.yaml")

    return process


def generate_token(identity: str, room: str) -> str:
    grant = VideoGrants(room_join=True, room=room)
    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(grant)
    )
    return token.to_jwt()


import requests  # ← add this at top of app.py

PC2_IP = "10.52.49.131"  # ← PC 2's IP address

def start_voice_handover(history: ConversationHistory):
    room_name = "billing-support-room"

    print("\n" + "="*50)
    print("📞 STARTING VOICE CALL WITH HUMAN AGENT")
    print("="*50)
    print(history.format_history())

    user_token = generate_token("user", room_name)
    agent_token = generate_token("human-agent", room_name)

    print(f"Room: {room_name}")

    # ✅ Auto send token to PC 2
    try:
        response = requests.post(
            f"http://{PC2_IP}:5000/start-agent",
            json={"token": agent_token},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Agent notified automatically!")
        else:
            print("❌ Failed to notify agent!")
            # Fallback — print token manually
            print(f"\n📋 Manual token: python human_agent.py {agent_token}")
    except Exception as e:
        print(f"❌ Could not reach PC 2: {e}")
        # Fallback — print token manually
        print(f"\n📋 Manual token: python human_agent.py {agent_token}")

    return user_token, room_name

async def voice_session(user_token: str, room_name: str, history: ConversationHistory):
    from livekit import rtc
    from voice import VoiceListener

    room = rtc.Room()
    loop = asyncio.get_running_loop()
    handover_back = asyncio.Event()

    @room.on("data_received")
    def on_data(data: rtc.DataPacket, **kwargs):
        message = data.data.decode("utf-8")
        if message == "HANDOVER_TO_BOT":
            print("\n🔄 Human agent transferred conversation back to AI Bot!")
            loop.call_soon_threadsafe(handover_back.set)
        else:
            print(f"\n{message}")
            history.add("agent", message)

    @room.on("disconnected")
    def on_disconnected(**kwargs):
        print("\n⚠️ Disconnected from room.")
        loop.call_soon_threadsafe(handover_back.set)

    await room.connect(LIVEKIT_URL, user_token)
    print("✅ Connected! Speak into your microphone.")
    print("(AI is transcribing the conversation)\n")

    def on_user_speech(text: str):
        history.add("user", text)
        print(f"[DEBUG] Attempting to send to agent: '{text}'")
        future = asyncio.run_coroutine_threadsafe(
            room.local_participant.publish_data(
                f"User said: {text}".encode("utf-8")
            ),
            loop
        )
        try:
            future.result(timeout=5)
            print(f"[DEBUG] Successfully sent to agent! ✅")
        except Exception as e:
            print(f"[DEBUG] Failed to send to agent: {e} ❌")

    listener = VoiceListener(on_user_speech, label="User")
    listener.start()

    try:
        await handover_back.wait()
    except asyncio.CancelledError:
        pass
    finally:
        listener.stop()
        try:
            await room.disconnect()
        except Exception:
            pass

        # ✅ Print full context after handover
        print("\n🧠 FULL CONTEXT AI HAS:")
        print("="*50)
        for i, turn in enumerate(history.turns):
            print(f"{i+1}. [{turn['role'].upper()}]: {turn['content']}")
        print("="*50)
        print("\n✅ Back to AI Bot chat!\n")


def main():
    livekit_process = start_livekit()

    print("T-Mobile Billing FAQ RAG")
    print("Type your billing question, or 'exit' to quit.\n")

    history = ConversationHistory()
    failed_attempts = 0

    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            history.add("user", q)
            result = answer_billing_question(q, history)

            if result["type"] == "handover_check":
                failed_attempts += 1
                if result["message"]:
                    print(f"\nBot: {result['message']}")

                print("\nBot: I'm unable to fully resolve your issue.")
                print("Bot: Would you like me to connect you to a human agent? (yes/no)")
                confirm = input("You: ").strip().lower()

                if confirm in {"yes", "y"}:
                    user_token, room_name = start_voice_handover(history)
                    asyncio.run(voice_session(user_token, room_name, history))
                    failed_attempts = 0
                    print("Bot: Welcome back! How can I help you further?\n")
                else:
                    print("Bot: Okay! Please rephrase your question.\n")

            else:
                failed_attempts = 0
                history.add("assistant", result["message"])
                print(f"\nBot: {result['message']}\n")

    finally:
        if livekit_process:
            print("\n🛑 Stopping LiveKit server...")
            livekit_process.terminate()
            print("✅ LiveKit stopped!")


if __name__ == "__main__":
    main()