import os
import random
import numpy as np
import torch
import soundfile as sf
import librosa

DEFAULT_SEED_VALUE = 1337
SAMPLE_RATE_VALUE = 16000
MAX_DB_VALUE = 1e-9


def set_seed_for_training_purposes():
    """
    """

    random.seed(DEFAULT_SEED_VALUE)
    np.random.seed(DEFAULT_SEED_VALUE)
    torch.manual_seed(DEFAULT_SEED_VALUE)
    torch.cuda.manual_seed(DEFAULT_SEED_VALUE)


def normalize_audio_file(path: str, sample_rate: int = SAMPLE_RATE_VALUE) -> np.ndarray:
    """

    """

    x, r = sf.read(path, dtype="float", always_2d=False)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if r != sample_rate:
        x = librosa.resample(x, orig_sr=r, target_sr=sample_rate)

    m = np.max(np.abs(x)) + MAX_DB_VALUE

    if m > 1.0:
        x = x / m

    return x.astype("float32")


def save_wav(path: str, wav: np.ndarray, sample_rate: int = SAMPLE_RATE_VALUE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sample_rate)


def generate_random_segment(x: np.ndarray, number_of_samples: int) -> np.ndarray:
    """

    """
    if len(x) < number_of_samples:
        pad = np.zeros(number_of_samples, dtype=np.float32)
        pad[:len(x)] = x

        return pad

    start = random.randint(0, max(0, len(x) - number_of_samples))

    return x[start:start+number_of_samples]


def stft_mag(audio, n_fft=512, hop=128):
    """

    """
    win = torch.hann_window(n_fft, device=audio.device)

    X = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True)

    mag = torch.abs(X)
    phase = torch.angle(X)

    return mag.unsqueeze(1), phase, X


def istft(mag, phase, n_fft=512, hop=128):
    """
    mag: [B,1,F,Frames], phase [B,F,Frames]
    """

    win = torch.hann_window(n_fft, device=mag.device)
    Xc = torch.polar(mag.squeeze(1), phase)
    y = torch.istft(Xc, n_fft=n_fft, hop_length=hop, window=win)

    return y


def _rglob_wavs(root):
    root = os.path.abspath(root)

    out = []

    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                out.append(os.path.join(base, f))
    return out
