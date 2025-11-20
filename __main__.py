from colorama import init as colorama_init
from colorama import Fore, Style

from utils.config import config
from utils.datasets_downloader import download_all_datasets
from utils.prepare_data import generate_pairs_dataset

from datasets.audio_dataset import AudioDenoiseDataset
from torch.utils.data import DataLoader


def main():
    config.load(".env")

    colorama_init(autoreset=True, convert=True)

    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT +
          "            DENOISER - AUDIO CLEANING PROJECT            ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print()

    print(Fore.CYAN +
          "[INFO] Iniciando descarga de datasets...")

    datasets = download_all_datasets()

    print("\n" + Fore.GREEN + "[INFO] Descarga completa.")

    print(Fore.CYAN + "\n[INFO] Rutas finales detectadas:" + Style.RESET_ALL)

    for key, path in datasets.items():
        print(f"  - {key}: {path}")

    print("\n" + Fore.GREEN +
          "[OK] Todo listo para preparar datos." + Style.RESET_ALL)
    
    print(Fore.CYAN + "[INFO] Preparando dataset procesado (clean/noisy)...")
    generate_pairs_dataset()
    print(Fore.GREEN + "[OK] Datos procesados.\n")

    print(Fore.CYAN + "[INFO] Creando dataset Pytorch...")

    dataset = AudioDenoiseDataset(
        clean_dir="data/processed/clean",
        noisy_dir="data/processed/noisy"
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(Fore.GREEN + f"[OK] Dataset cargado. Total muestras: {len(dataset)}\n")

    """device = get_device()
    print(Fore.CYAN + f"[INFO] Usando dispositivo: {device}\n")"""


if __name__ == "__main__":
    main()
