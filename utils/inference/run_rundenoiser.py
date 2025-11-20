import torch
import librosa
import soundfile as sf
import numpy as np

from model.model_builder import UNetMini
from utils.config import config


SR = int(config["sample_rate"])
N_FFT = int(config["n_fft"])
HOP = int(config["hop_length"])


def stft_to_wav(mag, phase):
    stft = mag * np.exp(1j * phase)
    wav = librosa.istft(stft, hop_length=HOP)
    return wav


def denoise_file(model_path, input_wav, output_wav):
    model = UNetMini()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    wav, _ = librosa.load(input_wav, sr=SR, mono=True)
    stft = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)

    mag = np.abs(stft)
    phase = np.angle(stft)

    mag_tensor = torch.tensor(mag).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        pred_mag = model(mag_tensor).squeeze().cpu().numpy()

    out = stft_to_wav(pred_mag, phase)
    sf.write(output_wav, out, SR)
