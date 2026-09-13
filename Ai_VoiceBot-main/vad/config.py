"""Configuration constants for the VAD package.

Plain constants only — used by `vad_silero` and `stream_vad`.
"""

# Sampling rate (Hz) used for model inference and streaming I/O
SAMPLING_RATE: int = 16000

# Probability threshold (0.0-1.0) above which a frame is considered speech
VAD_THRESHOLD: float = 0.5

# Minimum speech length (milliseconds). Segments shorter than this are discarded.
MIN_SPEECH_MS: int = 200

# Merge silence gaps shorter than this (milliseconds) between segments.
MIN_SILENCE_MS: int = 150

# Frame/block size for streaming capture in milliseconds (used for buffering)
BLOCK_MS: int = 30

