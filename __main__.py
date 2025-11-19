from colorama import init as colorama_init
from colorama import Fore, Style

from utils.config import config
from utils.dataset_downloader import DatasetDownloader


def main():
    colorama_init(autoreset=True, convert=True)

    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT +
          "            DENOISER - AUDIO CLEANING PROJECT            ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print()

    config.load(".env")

    downloader = DatasetDownloader()
    downloader.run()
    # print("dataset_dir:", config.get("dataset_dir"))


if __name__ == "__main__":
    main()
