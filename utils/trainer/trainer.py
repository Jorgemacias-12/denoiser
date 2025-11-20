import torch
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0

        for noisy, clean in tqdm(dataloader, desc="Entrenando"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            pred = self.model(noisy)
            loss = F.l1_loss(pred, clean)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for noisy, clean in tqdm(dataloader, desc="Validando"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                pred = self.model(noisy)
                loss = F.l1_loss(pred, clean)

                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)
