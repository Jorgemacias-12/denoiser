import os 
import random 
import librosa
import soundfile as sf
import numpy as np
from glob import glob
from tqdm import tqdm
from utils.config import config

SAMPLE_RATE = int(config.get("sample_rate"))
CLIP_SECONDS = int(config.get("clip_seconds"))
SNR_MIN = int(config.get("snr_min"))
SNR_MAX = int(config.get("snr_max"))
TARGET_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

def load_and_process(path, sample_rate=SAMPLE_RATE):
    """Carga audio, lo convierte a mono y lo resamplea."""
    audio, sr = librosa.load(path, sr=sample_rate)
    
    #Audio más corto, pad
    if len(audio) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')

    #Audio más largo, crop aleatorio
    elif len(audio) > TARGET_SAMPLES:
        start = random.randint(0, len(audio) - TARGET_SAMPLES)
        audio = audio[start:start + TARGET_SAMPLES]
    
    return audio

def add_noise(clean, noise, snr_db):
    """Mezcla ruido a cierto SNR."""
    #Convertir SNR
    snr_linear = 10 ** (snr_db / 10)

    #Energía
    clean_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    #Calcular ganancia necesaria
    factor = np.sqrt(clean_power / (noise_power * snr_linear))
    noise_scaled = noise * factor
    
    return clean + noise_scaled

def main():
    clean_root = config["dataset_clean_dir"]
    noise_root = config["dataset_noise_dir"]

    clean_files = glob(os.path.join(clean_root, "**", "*.wav"), recursive=True)
    noise_files = glob(os.path.join(noise_root, "**", "*.wav"), recursive=True)

    out_clean = "data/processed/clean"
    out_noisy = "data/processed/noisy"
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    print(f"[INFO] Archivos clean encontrados: {len(clean_files)}")
    print(f"[INFO] Archivos noise encontrados: {len(noise_files)}")

    for clean_path in tqdm(clean_files, desc="Procesando clean"):
        clean_audio = load_and_process(clean_path)

        noise_path = random.choice(noise_files)
        noise_audio = load_and_process(noise_path)

        snr_db = random.uniform(SNR_MIN, SNR_MAX)
        noisy_audio = add_noise(clean_audio, noise_audio, snr_db)

        #Guardar con mismos nombres en carpetas separadas
        base = os.path.basename(clean_path)

        sf.write(os.path.join(out_clean, base), clean_audio, SAMPLE_RATE)
        sf.write(os.path.join(out_noisy, base), noisy_audio, SAMPLE_RATE)
    
    print("[OK] Datos procesados exitosamente")

if __name__ == "__main__":
    main()