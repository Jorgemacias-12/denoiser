import os
import random
from glob import glob

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.config import config

SAMPLE_RATE = int(config.get("sample_rate"))
CLIP_SECONDS = int(config.get("clip_seconds"))
SNR_MIN = int(config.get("snr_min"))
SNR_MAX = int(config.get("snr_max"))
TARGET_SAMPLES = SAMPLE_RATE * CLIP_SECONDS


def load_and_process(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Carga un archivo de audio, lo pasa a mono, lo remuestrea
    y lo ajusta a duración fija (CLIP_SECONDS).
    Devuelve un arreglo float32 de longitud TARGET_SAMPLES.
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)

    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    return audio.astype(np.float32)


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mezcla clean + noise ajustando el nivel del ruido para lograr
    un SNR objetivo (en dB).

    SNR(dB) = 10 * log10(P_signal / P_noise)
    """
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-10

    snr_linear = 10 ** (snr_db / 10.0)
    p_noise_target = p_signal / snr_linear
    scaling = np.sqrt(p_noise_target / p_noise)

    noise_scaled = noise * scaling
    noisy = clean + noise_scaled

    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
        clean = clean / max_val

    return noisy.astype(np.float32)

def main() -> None:
    print("[OK] Todo listo para preparar datos.")

    # Directorios crudos (ya descargados y descomprimidos)
    raw_clean_dir = "data/raw/clean"
    raw_noise_dir = "data/raw/noise"

    clean_files = sorted(glob(os.path.join(raw_clean_dir, "**", "*.wav"), recursive=True))
    noise_files = sorted(glob(os.path.join(raw_noise_dir, "**", "*.wav"), recursive=True))

    print(f"[INFO] Archivos clean encontrados: {len(clean_files)}")
    print(f"[INFO] Archivos noise encontrados: {len(noise_files)}")

    if not clean_files:
        raise RuntimeError("No se encontraron audios clean en data/raw/clean")
    if not noise_files:
        raise RuntimeError("No se encontraron audios noise en data/raw/noise")

    # Directorios de salida base
    out_clean_base = "data/processed/clean"
    out_noisy_base = "data/processed/noise"

    # Crear subcarpetas train/val/test
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(out_clean_base, split), exist_ok=True)
        os.makedirs(os.path.join(out_noisy_base, split), exist_ok=True)

    # ============================
    #  Split 60% train / 20% val / 20% test
    # ============================
    random.seed(42)
    random.shuffle(clean_files)

    n_total = len(clean_files)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val  # resto

    clean_train = clean_files[:n_train]
    clean_val = clean_files[n_train:n_train + n_val]
    clean_test = clean_files[n_train + n_val:]

    split_to_files = {
        "train": clean_train,
        "val": clean_val,
        "test": clean_test,
    }

    print(f"[INFO] Split -> train: {len(clean_train)}, val: {len(clean_val)}, test: {len(clean_test)}")

    # ============================
    #  Procesar y guardar
    # ============================

    for split_name, split_clean_files in split_to_files.items():
        out_clean = os.path.join(out_clean_base, split_name)
        out_noisy = os.path.join(out_noisy_base, split_name)

        print(f"[INFO] Procesando split '{split_name}' ({len(split_clean_files)} archivos)...")

        for clean_path in tqdm(split_clean_files):
            # Cargar clean
            clean_audio = load_and_process(clean_path)

            # Elegir ruido aleatorio
            noise_path = random.choice(noise_files)
            noise_audio = load_and_process(noise_path)

            # Mezclar con SNR aleatorio
            snr_db = random.uniform(SNR_MIN, SNR_MAX)
            noisy_audio = add_noise(clean_audio, noise_audio, snr_db)

            # Guardar con el mismo nombre de archivo
            base = os.path.basename(clean_path)
            sf.write(os.path.join(out_clean, base), clean_audio, SAMPLE_RATE)
            sf.write(os.path.join(out_noisy, base), noisy_audio, SAMPLE_RATE)

    print("[OK] Datos procesados exitosamente")


if __name__ == "__main__":
    main()
