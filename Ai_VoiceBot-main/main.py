import queue
import audioop
import sounddevice as sd
from vad import MicrophoneVAD
from ASR import ASR

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    rms = audioop.rms(bytes(indata), 2)
    
    q.put(bytes(indata))

def main():
    vad = MicrophoneVAD()
    asr = ASR()
    buffer = b""

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=4000,
        dtype='int16',
        channels=1,
        callback=audio_callback
    ):
        print("🎤 Listening for full sentences...")

        while True:
            data = q.get()

            if vad.is_speech(data):
                buffer += data

            elif vad.is_voice_break() and buffer:
                sentence = asr.process(buffer)
                buffer = b""

                if sentence and len(sentence.strip()) > 2:
                    print("🗣 User:", sentence)

if __name__ == "__main__":
    main()
