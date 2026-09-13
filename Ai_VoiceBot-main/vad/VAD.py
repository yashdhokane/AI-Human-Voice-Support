import numpy as np
import time


class MicrophoneVAD:
    def __init__(self, silence_threshold=100, silence_duration=1.5):
        """
        silence_threshold: energy level below which audio is considered silence
        silence_duration: seconds of silence to trigger voice break
        """
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.last_voice_time = time.time()

    def is_speech(self, audio_chunk: bytes) -> bool:
        audio = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.abs(audio).mean()

        if energy > self.silence_threshold:
            self.last_voice_time = time.time()
            return True
        return False

    def is_voice_break(self) -> bool:
        return (time.time() - self.last_voice_time) > self.silence_duration
