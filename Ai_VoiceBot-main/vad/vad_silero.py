"""Silero VAD wrapper for multiparty dialogue project.

Provides a `SileroVAD` class which loads the Silero VAD model via
`torch.hub` and exposes `get_speech_segments_from_file` to return
speech segments as (start_sec, end_sec) tuples.

This module is library-only (no CLI).
"""
from __future__ import annotations

from typing import List, Tuple
import os
import math

import numpy as np
import torch


class SileroVAD:
	"""Wrapper around Silero VAD.

	Parameters
	- sampling_rate: sample rate used for model and returned times (Hz).
	- threshold: probability threshold for speech activity.
	- min_speech_ms: minimum allowed speech segment length (ms).
	- min_silence_ms: merge gaps shorter than this (ms).
	"""

	# Cached model + utils for the process
	_model = None
	_utils = None

	def __init__(self, sampling_rate: int = 16000, threshold: float = 0.5,
				 min_speech_ms: int = 200, min_silence_ms: int = 150) -> None:
		self.sampling_rate = int(sampling_rate)
		self.threshold = float(threshold)
		self.min_speech_ms = int(min_speech_ms)
		self.min_silence_ms = int(min_silence_ms)

		# Ensure model loaded lazily
		if SileroVAD._model is None or SileroVAD._utils is None:
			SileroVAD._load_model()

	@classmethod
	def _load_model(cls) -> None:
		"""Load Silero VAD model and utils via torch.hub and cache them.

		Raises RuntimeError if the model cannot be loaded.
		"""
		try:
			model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', 
                               force_reload=False, 
                               skip_validation=True)
		except Exception as exc:
			raise RuntimeError(
				'Failed to load Silero VAD via torch.hub. Ensure internet access or the package is installed.'
			) from exc

		# utils is typically a tuple of helper functions
		try:
			# In silero examples utils contains several functions
			get_speech_timestamps = utils.get_speech_timestamps
		except Exception:
			# If utils is a sequence (model, utils) returned by hub, try unpacking common helpers
			try:
				get_speech_timestamps = utils[0]
			except Exception:
				raise RuntimeError('Could not locate get_speech_timestamps in silero utils')

		cls._model = model
		cls._utils = utils

	def _read_audio(self, path: str) -> np.ndarray:
		"""Read audio file and return mono float32 numpy array at `self.sampling_rate`.

		Tries silero utils `read_audio` first (if available), otherwise falls
		back to `torchaudio`.
		"""
		# Try silero utils read_audio if available
		utils = SileroVAD._utils
		if utils is not None and hasattr(utils, 'read_audio'):
			try:
				audio = utils.read_audio(path, sampling_rate=self.sampling_rate)
				audio = np.asarray(audio, dtype=np.float32)
				if audio.ndim > 1:
					audio = np.mean(audio, axis=0)
				return audio
			except Exception:
				pass

		# Fallback: use torchaudio
		try:
			import torchaudio

			waveform, sr = torchaudio.load(path)
			if sr != self.sampling_rate:
				waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sampling_rate)
			np_wav = waveform.numpy()
			if np_wav.ndim > 1:
				np_wav = np.mean(np_wav, axis=0)
			return np_wav.astype(np.float32)
		except Exception as exc:
			raise RuntimeError(f'Could not read audio file {path}: {exc}') from exc

	def get_speech_segments_from_file(self, path: str) -> List[Tuple[float, float]]:
		"""Return speech segments for the given audio file.

		Steps:
		- load audio (mono, at `self.sampling_rate`)
		- run Silero VAD to get raw speech timestamp candidates
		- merge segments separated by short silence (< `min_silence_ms`)
		- discard segments shorter than `min_speech_ms`

		Returns:
			List of (start_sec, end_sec) tuples (float seconds).
		"""
		if not os.path.isfile(path):
			raise FileNotFoundError(path)

		audio = self._read_audio(path)

		# Use silero's get_speech_timestamps if available in utils
		utils = SileroVAD._utils
		if utils is None:
			raise RuntimeError('Silero utils not loaded')

		# get_speech_timestamps is usually a function inside utils
		if hasattr(utils, 'get_speech_timestamps'):
			get_ts = utils.get_speech_timestamps
		else:
			# sometimes utils is a tuple/list with helpers
			try:
				# try common unpacking: (get_speech_timestamps, save_audio, read_audio, ...)
				get_ts = utils[0]
			except Exception:
				raise RuntimeError('Could not find get_speech_timestamps in silero utils')

		# Run VAD
		# silero expects a numpy array or torch tensor; pass numpy
		raw_timestamps = get_ts(audio, SileroVAD._model, sampling_rate=self.sampling_rate, threshold=self.threshold)

		# raw_timestamps is usually a list of dicts with 'start' and 'end' (samples)
		segments_samples: List[Tuple[int, int]] = []
		for item in raw_timestamps:
			# support both keys and attributes
			if isinstance(item, dict):
				s = int(item.get('start', 0))
				e = int(item.get('end', 0))
			else:
				# some variants return objects
				s = int(getattr(item, 'start', 0))
				e = int(getattr(item, 'end', 0))
			if e > s:
				segments_samples.append((s, e))

		if not segments_samples:
			return []

		# Convert to seconds
		segments = [(s / self.sampling_rate, e / self.sampling_rate) for s, e in segments_samples]

		# Merge segments separated by short silence
		merged: List[Tuple[float, float]] = []
		min_silence_sec = self.min_silence_ms / 1000.0
		for start, end in segments:
			if not merged:
				merged.append((start, end))
				continue
			prev_start, prev_end = merged[-1]
			gap = start - prev_end
			if gap <= min_silence_sec:
				# merge
				merged[-1] = (prev_start, max(prev_end, end))
			else:
				merged.append((start, end))

		# Discard segments shorter than min_speech_ms
		min_speech_sec = self.min_speech_ms / 1000.0
		final: List[Tuple[float, float]] = []
		for start, end in merged:
			if (end - start) + 1e-12 >= min_speech_sec:
				final.append((float(start), float(end)))

		return final


__all__ = ["SileroVAD"]

