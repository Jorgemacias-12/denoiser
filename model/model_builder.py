from torch.nn import Module, ModuleList, ReLU, Sequential, Conv2d, MaxPool2d, Sigmoid
import torch.nn.functional as F
from torch import cat


class UNet(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """

    def block(c_in, c_out):
        return Sequential(
            Conv2d(c_in, c_out, 3, padding=1), ReLU(inplace=True),
            Conv2d(c_out, c_out, 3, padding=1), ReLU(inplace=True),
        )

    def __init__(self, in_ch=1, base=32, depth=5):
        super().__init__()

        self.depth = depth

        # Encoder section
        self.down, self.pools = ModuleList(), ModuleList()

        ch = in_ch

        for d in range(depth):
            out_ch = base * (2 ** d)
            self.down.append(self.block(ch, out_ch))
            self.pools.append(MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch

        # Bottleneck section
        self.bottleneck = self.block(ch, ch * 2)
        ch *= 2

        # Decoder section
        self.up = ModuleList()
        self.dec = ModuleList()

        # Head
        self.head = Sequential(
            Conv2d(ch, 1, kernel_size=1), Sigmoid())

    def forward(self, mag):
        B, C, H0, W0 = mag.shape

        m = 2 ** self.depth
        pad_h = (-H0) % m
        pad_w = (-W0) % m
        if pad_h or pad_w:
            # (left, right, top, bottom)
            mag = F.pad(mag, (0, pad_w, 0, pad_h))

        feats, x = [], mag
        for enc, pool in zip(self.down, self.pools):
            x = enc(x)
            feats.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder con alineación exacta a cada skip
        for up, dec, skip in zip(self.up, self.dec, reversed(feats)):
            x = up(x)
            # Fuerza tamaño EXACTO al del skip (evita off-by-one)
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = cat([x, skip], dim=1)
            x = dec(x)

        mask = self.head(x)  # [B,1,F',T']

        # ---- Recorta el pad para volver a [B,1,H0,W0] ----
        if pad_h or pad_w:
            mask = mask[..., :H0, :W0]

        return mask
