# model/model_builder.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple

class ConvBlock(nn.Module):
    """
    Bloque convolucional básico: (Conv2d -> BN -> PReLU) x 2
    Mantiene tamaño espacial si padding = 1 y kernel_size = 3.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Downsample(nn.Module):
    """
    Reduce por factor 2 usando conv stride=2 (mejor comportamiento en audio que maxpool en algunos casos).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Upsample(nn.Module):
    """
    Upsample mediante ConvTranspose2d (learnable) seguido de conv-block.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # ConvTranspose2d doubles spatial dims when stride=2, kernel_size=2
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # En caso de mismatch por redondeos, recortar/pad si es necesario
        if x.size(-2) != skip.size(-2) or x.size(-1) != skip.size(-1):
            x = nn.functional.interpolate(x, size=(skip.size(-2), skip.size(-1)), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSpectral(nn.Module):
    """
    UNet 2D para espectrogramas.

    Args:
        in_channels: 1 (magnitud) normalmente
        base_channels: número de canales en primer nivel (e.g. 32)
        depth: niveles de encoder/decoder (recomendado 5)
        mask_output: si True la red predice una máscara multiplicativa (sigmoid),
                     si False predice directamente la magnitud (ReLU o linear).
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, depth: int = 5, mask_output: bool = True):
        super().__init__()
        assert depth >= 3 and depth <= 7, "depth típico entre 3 y 7"
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.mask_output = mask_output

        # Encoder
        enc_blocks: List[nn.Module] = []
        down_blocks: List[nn.Module] = []

        ch = base_channels
        enc_blocks.append(ConvBlock(in_channels, ch))
        for i in range(1, depth):
            down_blocks.append(Downsample(ch, ch * 2))
            ch *= 2
            enc_blocks.append(ConvBlock(ch, ch))

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)

        # Bottleneck (after last downsample)
        self.bottleneck = ConvBlock(ch, ch * 2)
        mid_ch = ch * 2

        # Decoder
        up_blocks: List[nn.Module] = []
        for i in range(depth - 1):
            up_blocks.append(Upsample(mid_ch, mid_ch // 2))
            mid_ch = mid_ch // 2

        self.up_blocks = nn.ModuleList(up_blocks)

        # Final conv to map to single-channel magnitude or mask
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_ch, self.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.PReLU(),
            nn.Conv2d(self.base_channels, 1, kernel_size=1)
        )

        # If mask_output, use sigmoid in forward; otherwise ReLU or identity (we'll return raw)
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, F, T]  (F = freq bins, T = time frames)
        return:
            if mask_output: mask in [0,1], same shape as x, so caller multiplies |STFT|*mask
            else: predicted magnitude (non-negative) — we apply relu to be safe
        """
        enc_feats: List[torch.Tensor] = []
        out = x
        # encoder path
        for i, enc in enumerate(self.enc_blocks):
            out = enc(out)
            enc_feats.append(out)
            if i < len(self.down_blocks):
                out = self.down_blocks[i](out)

        # bottleneck
        out = self.bottleneck(out)

        # decoder path
        for i, up in enumerate(self.up_blocks):
            skip = enc_feats[-(i + 1)]
            out = up(out, skip)

        out = self.final_conv(out)  # [B, 1, F, T]

        if self.mask_output:
            mask = torch.sigmoid(out)
            return mask
        else:
            # raw magnitude prediction, ensure non-negative
            return torch.relu(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary_example():
    """
    Ejemplo: instancia y print de parámetros.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSpectral(in_channels=1, base_channels=32, depth=5, mask_output=True).to(device)
    nparams = count_parameters(model)
    print(f"UNetSpectral params: {nparams:,}")
    return model
