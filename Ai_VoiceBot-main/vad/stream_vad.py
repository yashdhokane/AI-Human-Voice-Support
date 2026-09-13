"""Streaming VAD using SileroVAD and the default microphone.

Provides `MicrophoneVAD` which captures audio with `sounddevice`, buffers
30 ms frames, runs Silero VAD on a rolling window, and calls back when
speech starts/ends.

This module is library-only.
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .vad_silero import SileroVAD


class MicrophoneVAD:
	"""Simple microphone-based streaming VAD.

	Parameters
	- vad: an instance of `SileroVAD` (if None, a default is created).
	- on_speech_start: Callable[[float], None] called with wall-clock timestamp (sec).
	- on_speech_end: Callable[[float], None] called with wall-clock timestamp (sec).
	- sample_rate: sampling rate in Hz (default 16000).
	- frame_ms: frame size in milliseconds (default 30ms).
	- window_ms: rolling window size passed to VAD (default 1000ms).
	- threshold: VAD probability threshold passed to SileroVAD.
	"""

	def __init__(self,
				 vad: Optional[SileroVAD] = None,
				 on_speech_start: Optional[Callable[[float], None]] = None,
				 on_speech_end: Optional[Callable[[float], None]] = None,
				 sample_rate: int = 16000,
				 frame_ms: int = 30,
				 window_ms: int = 1000,
				 threshold: float = 0.5):
		self.vad = vad or SileroVAD(sampling_rate=sample_rate, threshold=threshold)
		self.on_speech_start = on_speech_start
		self.on_speech_end = on_speech_end

		self.sample_rate = int(sample_rate)
		self.frame_ms = int(frame_ms)
		self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))
		self.window_ms = int(window_ms)
		self.window_size = int(self.sample_rate * (self.window_ms / 1000.0))

		# Internal state
		self._stream: Optional[sd.InputStream] = None
		self._buffer = deque()  # holds numpy arrays (frames)
		self._buffer_lock = threading.Lock()
		self._worker: Optional[threading.Thread] = None
		self._stop_event = threading.Event()

		# Sample accounting
		self._total_samples = 0  # total samples received since start
		self._stream_start_time: Optional[float] = None

		# Speech tracking
		self._speech_active = False
		self._last_reported_end_sample = -1
		# current speech assembly (global sample indices and audio)
		self._current_speech_start_global: Optional[int] = None
		self._current_speech_last_end_global: Optional[int] = None
		self._current_speech_audio: Optional[np.ndarray] = None

	def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
		# indata shape: (frames, channels)
		if status:
			# don't raise in callback; just ignore status
			pass

		# Ensure mono
		arr = indata
		if arr.ndim > 1:
			arr = np.mean(arr, axis=1)
		arr = arr.astype(np.float32)

		with self._buffer_lock:
			self._buffer.append(arr.copy())
			# keep buffer length bounded by window_size
			total = sum(x.shape[0] for x in self._buffer)
			while total > self.window_size:
				removed = self._buffer.popleft()
				total -= removed.shape[0]

		self._total_samples += frames

	def _worker_loop(self):
		utils = SileroVAD._utils
		model = SileroVAD._model
		if utils is None or model is None:
			raise RuntimeError('SileroVAD model/utils not loaded')

		# helper to build current window array and compute its global sample start
		while not self._stop_event.is_set():
			with self._buffer_lock:
				if not self._buffer:
					window = np.zeros(0, dtype=np.float32)
				else:
					window = np.concatenate(list(self._buffer)).astype(np.float32)
				window_len = window.shape[0]

				# global start sample for this window
				global_start_sample = max(0, self._total_samples - window_len)

			if window_len >= 1:
				# call silero utils directly using cached model/utils
				# get_speech_timestamps expects numpy array and returns list of dicts
				get_ts = None
				if hasattr(utils, 'get_speech_timestamps'):
					get_ts = utils.get_speech_timestamps
				else:
					try:
						get_ts = utils[0]
					except Exception:
						raise RuntimeError('Could not locate get_speech_timestamps')

				try:
					results = get_ts(window, model, sampling_rate=self.sample_rate, threshold=self.vad.threshold)
				except Exception:
					results = []

				# Determine if any speech present in window
				if results:
					# compute earliest start and latest end in global sample indices
					starts = [int(item.get('start', 0)) if isinstance(item, dict) else int(getattr(item, 'start', 0)) for item in results]
					ends = [int(item.get('end', 0)) if isinstance(item, dict) else int(getattr(item, 'end', 0)) for item in results]
					first_start = min(starts)
					last_end = max(ends)

					first_start_global = global_start_sample + first_start
					last_end_global = global_start_sample + last_end

					ts_start = (first_start_global / self.sample_rate) + (self._stream_start_time or 0.0)
					ts_end = (last_end_global / self.sample_rate) + (self._stream_start_time or 0.0)

					if not self._speech_active:
						# begin a new speech segment and capture audio slice
						self._speech_active = True
						self._current_speech_start_global = first_start_global
						self._current_speech_last_end_global = last_end_global
						# slice from window (indices relative to window)
						self._current_speech_audio = window[first_start:last_end].copy()
						if self.on_speech_start:
							try:
								self.on_speech_start(ts_start)
							except Exception:
								pass
					else:
						# speech already active -> append any newly detected tail
						prev_end = int(self._current_speech_last_end_global or 0)
						if last_end_global > prev_end:
							append_start_global = max(prev_end, first_start_global)
							append_start_in_window = int(max(0, append_start_global - global_start_sample))
							append_end_in_window = int(last_end)
							if append_end_in_window > append_start_in_window:
								append = window[append_start_in_window:append_end_in_window]
								if self._current_speech_audio is None:
									self._current_speech_audio = append.copy()
								else:
									self._current_speech_audio = np.concatenate([self._current_speech_audio, append])
						# update last end
						self._current_speech_last_end_global = max(self._current_speech_last_end_global or 0, last_end_global)

					# update last reported end sample for deduplication
					self._last_reported_end_sample = max(self._last_reported_end_sample, last_end_global)
				else:
					# no speech in current window
					if self._speech_active:
						# report end using assembled speech audio
						end_sample = int(self._current_speech_last_end_global or self._total_samples)
						ts_end = (end_sample / self.sample_rate) + (self._stream_start_time or 0.0)
						speech_audio = None
						if self._current_speech_audio is not None:
							speech_audio = self._current_speech_audio.copy()
						# reset state
						self._speech_active = False
						self._current_speech_start_global = None
						self._current_speech_last_end_global = None
						self._current_speech_audio = None
						self._last_reported_end_sample = -1
						if self.on_speech_end:
							try:
								# pass timestamp and audio numpy array
								self.on_speech_end(ts_end, speech_audio)
							except Exception:
								pass

			# sleep for one frame duration
			time.sleep(self.frame_ms / 1000.0)

	def start(self) -> None:
		"""Start capturing audio and running VAD.

		The `on_speech_start` / `on_speech_end` callbacks will be called with
		wall-clock timestamps (time.time()).
		"""
		if self._stream is not None:
			return

		self._stop_event.clear()
		self._buffer.clear()
		self._total_samples = 0
		self._speech_active = False
		self._last_reported_end_sample = -1
		self._stream_start_time = time.time()

		self._stream = sd.InputStream(samplerate=self.sample_rate,
									  channels=1,
									  dtype='float32',
									  blocksize=self.frame_size,
									  callback=self._audio_callback)
		self._stream.start()

		self._worker = threading.Thread(target=self._worker_loop, daemon=True)
		self._worker.start()

	def stop(self) -> None:
		"""Stop capturing and processing audio."""
		self._stop_event.set()
		if self._worker is not None:
			self._worker.join(timeout=1.0)
			self._worker = None

		if self._stream is not None:
			try:
				self._stream.stop()
				self._stream.close()
			except Exception:
				pass
			self._stream = None


__all__ = ["MicrophoneVAD"]

