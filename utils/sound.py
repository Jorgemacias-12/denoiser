import soundfile as sf
import os
from colorama import Fore
from tqdm import tqdm

from utils.file import remove_zone_identifier, make_file_writable, ensure_file_unlocked


def convert_flac_to_wav(root_dir: str, delete_flac=False):

    print(Fore.CYAN + "[INFO] Convirtiendo FLAC → WAV...")

    flac_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".flac"):
                flac_files.append(os.path.join(subdir, file))

    for flac_path in tqdm(flac_files, desc="Convirtiendo", unit="archivo"):

        wav_path = flac_path[:-5] + ".wav"

        # FIX 1: quitar bloqueos de Windows
        remove_zone_identifier(flac_path)

        # FIX 2: permisos write
        make_file_writable(flac_path)

        # FIX 3: reintentar si Windows bloquea
        if not ensure_file_unlocked(flac_path):
            continue

        try:
            audio, sr = sf.read(flac_path)
            sf.write(wav_path, audio, sr)

            if delete_flac:
                make_file_writable(flac_path)
                try:
                    os.remove(flac_path)
                except PermissionError:
                    print(
                        Fore.RED + f"[ERROR] No se pudo borrar {flac_path}")

        except Exception as e:
            print(
                Fore.RED + f"[ERROR] Falló {flac_path}: {e}")

    print(Fore.GREEN + "[INFO] Conversión terminada.")
