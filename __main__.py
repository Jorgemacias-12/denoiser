from colorama import init as colorama_init
from colorama import Fore, Style


def main():
    colorama_init(autoreset=True, convert=True)

    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT +
          "            DENOISER - AUDIO CLEANING PROJECT            ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print()


if __name__ == "__main__":
    main()
