import torch
from colorama import Fore, Style


def get_device(verbose: bool = True):
    """
    Detecta automáticamente el dispositivo disponible:
    - CUDA (GPU NVIDIA)
    - MPS (GPU Apple Silicon)
    - CPU
    
    Args:
        verbose: Si True, imprime el estado detectado.
        
    Returns:
        torch.device: El dispositivo seleccionado.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(Fore.GREEN + f"[INFO] GPU CUDA disponible: {torch.cuda.get_device_name(0)}")
        return device

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(Fore.GREEN + "[INFO] GPU MPS (Apple Silicon) disponible")
        return device

    else:
        if verbose:
            print(Fore.YELLOW + "[WARN] No hay GPU disponible, usando CPU.")
        return torch.device("cpu")
