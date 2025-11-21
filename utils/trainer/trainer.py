import torch
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def _forward_step(self, noisy, clean):
        """
        Hace:
          - forward del modelo
          - ajuste de tamaño de la máscara (si hace falta)
          - aplica la máscara al espectrograma ruidoso
          - calcula la loss L1 contra el clean
        """
        # 1) Predicción de máscara
        mask = self.model(noisy)  # [B, 1, f', t']

        # 2) Ajustar tamaño de la máscara si no coincide con el target
        if mask.shape[-2:] != clean.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=clean.shape[-2:],      # (F, T) del espectrograma original
                mode="bilinear",
                align_corners=False,
            )

        # 3) Aplicar máscara al espectrograma ruidoso
        est_clean = mask * noisy  # [B, 1, F, T]

        # 4) Loss L1 entre espectrograma limpio estimado y real
        loss = F.l1_loss(est_clean, clean)
        return loss

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0

        for noisy, clean in tqdm(dataloader, desc="Entrenando"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()

            loss = self._forward_step(noisy, clean)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for noisy, clean in tqdm(dataloader, desc="Validando"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                loss = self._forward_step(noisy, clean)
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)
