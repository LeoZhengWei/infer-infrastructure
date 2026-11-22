"""Kaldi-compatible 80-dim fbank + utterance CMN.

Matches the 3D-Speaker / torchaudio.compliance.kaldi.fbank settings used by
CAM++: 16 kHz, 25 ms / 10 ms, Povey window, preemphasis 0.97, mel 20 Hz..Nyquist,
log floor at float epsilon, then subtract the per-utterance mean.
"""

import math
import wave

import numpy as np
import torch


def load_wav(path: str) -> torch.Tensor:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getframerate() != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {w.getframerate()}")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    return torch.from_numpy(pcm / 32768.0)


def _mel_banks(num_bins: int, fft_len: int, sample_rate: int, low_hz: float, high_hz: float) -> torch.Tensor:
    num_fft_bins = fft_len // 2  # Kaldi drops the Nyquist bin
    if high_hz <= 0.0:
        high_hz = 0.5 * sample_rate
    mel_low = 1127.0 * math.log(1.0 + low_hz / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_hz / 700.0)
    mels = torch.linspace(mel_low, mel_high, num_bins + 2)
    fft_bin_width = sample_rate / fft_len
    fft_mels = 1127.0 * torch.log1p(torch.arange(num_fft_bins, dtype=torch.float32) * fft_bin_width / 700.0)
    banks = torch.zeros(num_bins, num_fft_bins)
    for i in range(num_bins):
        left, center, right = mels[i], mels[i + 1], mels[i + 2]
        up = (fft_mels - left) / (center - left)
        down = (right - fft_mels) / (right - center)
        banks[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return banks


_BANKS = None


def fbank(wav: torch.Tensor, sample_rate: int = 16000, num_mel: int = 80) -> torch.Tensor:
    """wav: (samples,) in [-1, 1]. Returns (T, num_mel) float32, mean-normalized."""
    global _BANKS
    if wav.dim() != 1:
        raise ValueError("wav must be 1-D")
    frame_len = int(sample_rate * 0.025)
    frame_shift = int(sample_rate * 0.010)
    if wav.numel() < frame_len:
        raise ValueError("audio shorter than one frame")
    frames = wav.unfold(0, frame_len, frame_shift).to(torch.float32)
    frames = frames - frames.mean(dim=1, keepdim=True)
    pre = frames.clone()
    pre[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    pre[:, 0] = frames[:, 0] - 0.97 * frames[:, 0]
    n = torch.arange(frame_len, dtype=torch.float32)
    window = torch.pow(0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (frame_len - 1)), 0.85)
    pre = pre * window
    fft_len = 1
    while fft_len < frame_len:
        fft_len *= 2
    padded = torch.nn.functional.pad(pre, (0, fft_len - frame_len))
    power = torch.fft.rfft(padded).abs().pow(2)[:, : fft_len // 2]
    if _BANKS is None or _BANKS.shape[0] != num_mel:
        _BANKS = _mel_banks(num_mel, fft_len, sample_rate, 20.0, 0.0)
    eps = torch.finfo(torch.float32).eps
    feat = torch.log(torch.clamp(power @ _BANKS.T, min=eps))
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.contiguous()
