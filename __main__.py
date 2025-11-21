from colorama import init as colorama_init
from colorama import Fore, Style

from utils.config import config
from utils.datasets_downloader import download_all_datasets
from utils.prepare_data import main as prepare_data
from utils.device import get_device

from model.model_builder import UNetSpectral
from utils.trainer.trainer import Trainer

from datasets.audio_dataset import AudioDenoiseDataset
from torch.utils.data import DataLoader

import torch
import os


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

    prepare_data()

    print(Fore.GREEN + "[OK] Datos procesados.\n")

    print(Fore.CYAN + "[INFO] Creando dataset Pytorch...")

    dataset_train = AudioDenoiseDataset(
        clean_dir="data/processed/clean",
        noisy_dir="data/processed/noise", 
        split ="train"
    )

    dataset_val = AudioDenoiseDataset(
        clean_dir="data/processed/clean",
        noisy_dir="data/processed/noise", 
        split ="val"
    )

    dataset_test = AudioDenoiseDataset(
        clean_dir="data/processed/clean",
        noisy_dir="data/processed/noise", 
        split ="test"
    )

    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False)

    print(Fore.GREEN + f"[OK] Train samples: {len(dataset_train)}")
    print(Fore.GREEN + f"[OK] Val samples: {len(dataset_val)}\n")
    print(Fore.CYAN +  f"[OK] Val samples: {len(dataset_test)}\n")

    device = get_device()
    print(Fore.CYAN + f"[INFO] Usando dispositivo: {device}\n")

    print(Fore.CYAN + "[INFO] Creando modelo UNetSpectral...")

    model = UNetSpectral(
        in_channels=1,
        base_channels=32,
        depth=5,
        mask_output=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(Fore.GREEN + f"[OK] Modelo creado con {total_params:,} parámetros.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = Trainer(model, optimizer, device)


    EPOCHS = int(config.get("epochs", 20))
    print(Fore.CYAN + f"[INFO] Entrenando por {EPOCHS} épocas...\n")

    best_val = 9999
    ckpt_dir = config.get("ckpt_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):

        print(Fore.YELLOW + f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)

        print(Fore.GREEN + f"[INFO] Train Loss: {train_loss:.4f}")
        print(Fore.CYAN + f"[INFO] Val   Loss: {val_loss:.4f}\n")

        # Guardar el mejor modelo
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(ckpt_dir, "denoiser_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(Fore.MAGENTA + f"[OK] Checkpoint actualizado: {ckpt_path}")

    print(Fore.GREEN + Style.BRIGHT + "\n[DONE] Entrenamiento completado sin errores.")
    print(Fore.GREEN + "[INFO] Modelo guardado en checkpoints.")

if __name__ == "__main__":
    main()
