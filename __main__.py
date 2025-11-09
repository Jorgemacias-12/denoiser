from colorama import init, Fore
from torch import cuda
import torch

from scripts.generate_train_data import prepare_mini_dataset

init(autoreset=True)


def main():
    """
    Función principal que detecta y configura el dispositivo a usar (CPU o GPU)
    """
    # Check if CUDA (NVIDIA GPU) is available else use CPU
    if cuda.is_available():
        device_to_use = "cuda"
        device_name = cuda.get_device_name(0)
        print(Fore.GREEN + "✓ Se usará GPU para la ejecución")
        print(Fore.CYAN + f"GPU detectada: {device_name}")
    else:
        device_to_use = "cpu"
        device_name = "CPU"
        print(Fore.YELLOW + "⚠ No se detectó GPU compatible, se usará CPU")
        print(Fore.CYAN + f"Dispositivo: {device_name}")

    # Apply device to use
    torch.set_default_device(device_to_use)
    print(Fore.MAGENTA + f"Dispositivo configurado: {device_to_use}")

    # Create training data
    prepare_mini_dataset()

    # Train model

    # Save Model


if __name__ == "__main__":
    main()
