from colorama import init as colorama_init
from colorama import Fore, Style

from utils.config import config
from utils.datasets_downloader import download_all_datasets


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


if __name__ == "__main__":
    main()
