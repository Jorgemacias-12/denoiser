import os
import torch
import numpy as np
import librosa
from glob import glob
from torch.utils.data import Dataset

from utils.config import config

SAMPLE_RATE = int(config.get("sample_rate"))
N_FFT = int(config.get("n_fft"))
HOP_LEN = int(config.get("hop_length"))


class AudioDenoiseDataset(Dataset):
    def __init__(
        self,
        clean_dir="data/processed/clean",
        noisy_dir="data/processed/noisy",
        split=None  # 'train', 'val', 'test' o None
    ):
        # Si se especifica split, añadimos la subcarpeta
        if split is not None:
            clean_dir = os.path.join(clean_dir, split)
            noisy_dir = os.path.join(noisy_dir, split)

        self.clean_files = sorted(glob(os.path.join(clean_dir, "*.wav")))
        self.noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))

    def __len__(self):
        return len(self.clean_files)

    def load_stft(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LEN)
        mag = np.abs(stft)
        return mag

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = np.random.choice(self.noisy_files)

        clean_mag = self.load_stft(clean_path)
        noisy_mag = self.load_stft(noisy_path)

        clean_mag = torch.tensor(clean_mag, dtype=torch.float32)
        noisy_mag = torch.tensor(noisy_mag, dtype=torch.float32)

        return noisy_mag.unsqueeze(0), clean_mag.unsqueeze(0)
