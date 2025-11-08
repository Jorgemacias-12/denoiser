import os
import sys
import math
import time
import glob
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import soundfile as sf

# -----------------------------------------------------------------------------
# 0) Rutas locales
# -----------------------------------------------------------------------------

# Raíz del proyecto = carpeta donde está este script
PROJ = os.path.abspath(os.path.dirname(__file__))

# Árbol de directorios que usará el proyecto
TREE = [
    os.path.join(PROJ, "configs"),
    os.path.join(PROJ, "data", "clean"),
    os.path.join(PROJ, "data", "noise"),
    os.path.join(PROJ, "denoiser", "models"),
    os.path.join(PROJ, "denoiser"),
    os.path.join(PROJ, "checkpoints"),
    os.path.join(PROJ, "outputs"),
    os.path.join(PROJ, "outputs", "samples"),
    os.path.join(PROJ, "outputs", "denoised"),
    os.path.join(PROJ, "inputs"),
    os.path.join(PROJ, "speechcommands"),
]
for d in TREE:
    os.makedirs(d, exist_ok=True)

print("Proyecto en:", PROJ)
print("Python:", sys.version)
print("Torch :", torch.__version__, "| CUDA disponible:", torch.cuda.is_available())
try:
    print("torchaudio:", torchaudio.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass

# -----------------------------------------------------------------------------
# 1) Utilidades de audio
# -----------------------------------------------------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Carga WAV a mono y remuestrea si es necesario, normalizando si hiciera falta."""
    x, r = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if r != sr:
        import librosa  # carga perezosa
        x = librosa.resample(x, orig_sr=r, target_sr=sr)
    # normaliza leve si es necesario
    m = np.max(np.abs(x)) + 1e-9
    if m > 1.0:
        x = x / m
    return x.astype("float32")

def save_wav(path: str, wav: np.ndarray, sr: int = 16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)

def segment_random(x: np.ndarray, n_samples: int) -> np.ndarray:
    if len(x) < n_samples:
        pad = np.zeros(n_samples, dtype=np.float32)
        pad[:len(x)] = x
        return pad
    start = random.randint(0, max(0, len(x) - n_samples))
    return x[start:start+n_samples]

# -----------------------------------------------------------------------------
# 2) STFT helpers
# -----------------------------------------------------------------------------

def stft_mag(audio, n_fft=512, hop=128):
    """
    audio: [B, T] float32
    return: mag [B,1,F,Frames], phase [B,F,Frames], complex [B,F,Frames]
    """
    win = torch.hann_window(n_fft, device=audio.device)
    X = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True)
    mag = torch.abs(X)
    phase = torch.angle(X)
    return mag.unsqueeze(1), phase, X

def istft(mag, phase, n_fft=512, hop=128):
    """
    mag: [B,1,F,Frames], phase [B,F,Frames]
    """
    win = torch.hann_window(n_fft, device=mag.device)
    Xc = torch.polar(mag.squeeze(1), phase)  # to complex
    y = torch.istft(Xc, n_fft=n_fft, hop_length=hop, window=win)
    return y

# -----------------------------------------------------------------------------
# 3) Dataset de pares (clean + noise) con mezcla SNR aleatoria
# -----------------------------------------------------------------------------

class NoisyPairDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sr=16000, clip_seconds=3, snr_min=0, snr_max=20):
        self.clean = sorted([p for p in _rglob_wavs(clean_dir)])
        self.noise = sorted([p for p in _rglob_wavs(noise_dir)])
        assert len(self.clean)>0, f"No hay WAVs limpios en {clean_dir}."
        assert len(self.noise)>0, f"No hay WAVs de ruido en {noise_dir}."
        self.sr = sr
        self.clip = int(sr*clip_seconds)
        self.snr_min, self.snr_max = snr_min, snr_max

    def __len__(self): return len(self.clean)

    def __getitem__(self, idx):
        clean = load_wav(self.clean[idx], self.sr)
        noise = load_wav(random.choice(self.noise), self.sr)
        clean = segment_random(clean, self.clip)
        noise = segment_random(noise, self.clip)

        # Mezcla con SNR aleatorio
        snr_db = random.uniform(self.snr_min, self.snr_max)
        s_pow = (clean**2).mean() + 1e-9
        n_pow = (noise**2).mean() + 1e-9
        k = np.sqrt(s_pow / (n_pow * (10**(snr_db/10))))
        noisy = clean + k*noise

        return torch.from_numpy(noisy), torch.from_numpy(clean)

def _rglob_wavs(root):
    root = os.path.abspath(root)
    out = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                out.append(os.path.join(base, f))
    return out

# -----------------------------------------------------------------------------
# 4) U-Net para espectrogramas
# -----------------------------------------------------------------------------

def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
    )

class UNetSpec(nn.Module):
    def __init__(self, in_ch=1, base=32, depth=5):
        super().__init__()
        self.depth = depth

        # Encoder
        self.down, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            out_ch = base * (2 ** d)
            self.down.append(block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = block(ch, ch * 2)
        ch *= 2

        # Decoder
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for d in reversed(range(depth)):
            out_ch = base * (2 ** d)
            self.up.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch, out_ch, kernel_size=1),
                )
            )
            self.dec.append(block(out_ch * 2, out_ch))
            ch = out_ch

        # Head
        self.head = nn.Sequential(nn.Conv2d(ch, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, mag):           # mag: [B,1,F,T]
        B, C, H0, W0 = mag.shape

        # ---- Pad a múltiplo de 2**depth (solo derecha/abajo) ----
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
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        mask = self.head(x)  # [B,1,F',T']

        # ---- Recorta el pad para volver a [B,1,H0,W0] ----
        if pad_h or pad_w:
            mask = mask[..., :H0, :W0]

        return mask

# -----------------------------------------------------------------------------
# 5) Descarga/filtrado de SpeechCommands + creación de dataset limpio/ruido
# -----------------------------------------------------------------------------

def prepare_mini_dataset(cfg, force=False):
    """
    Crea CLEAN y NOISE sintético a partir de SpeechCommands (one, two, three).
    Si ya existen archivos, no rehace el proceso salvo que force=True.
    """
    SR = cfg["sample_rate"]
    clean_train = os.path.join(PROJ, "data", "clean", "train")
    clean_val   = os.path.join(PROJ, "data", "clean", "val")
    clean_test  = os.path.join(PROJ, "data", "clean", "test")
    noise_train = os.path.join(PROJ, "data", "noise", "train")
    noise_val   = os.path.join(PROJ, "data", "noise", "val")
    noise_test  = os.path.join(PROJ, "data", "noise", "test")
    for d in [clean_train, clean_val, clean_test, noise_train, noise_val, noise_test]:
        os.makedirs(d, exist_ok=True)

    # Si ya hay WAVs en clean y noise, evitamos rehacer
    if any(os.listdir(clean_train)) and any(os.listdir(noise_train)) and not force:
        print("Dataset limpio/ruido ya existe. Saltando creación (usa force=True para rehacer).")
        return

    # 5.1) Descargar SpeechCommands (versión de torchaudio)
    speechcommands_path = os.path.join(PROJ, "speechcommands")
    os.makedirs(speechcommands_path, exist_ok=True)
    print("Descargando/leyendo SPEECHCOMMANDS...")
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=speechcommands_path, download=True)

    TARGET_LABELS = ['one', 'two', 'three']
    filtered_indices = []
    for i in range(len(ds)):
        try:
            _, _, label, _, _ = ds[i]
            if label in TARGET_LABELS:
                filtered_indices.append(i)
        except Exception as e:
            print(f"Error al procesar la muestra {i}: {e}")

    print(f"Total muestras filtradas ({', '.join(TARGET_LABELS)}): {len(filtered_indices)}")

    # 5.2) Split 60/20/20 sobre filtradas
    N = len(filtered_indices)
    idx = list(range(N))
    random.shuffle(idx)
    train_cut = int(0.6*N); val_cut = int(0.8*N)
    splits = {"train": idx[:train_cut], "val": idx[train_cut:val_cut], "test": idx[val_cut:]}

    # 5.3) Guardar CLEAN (a 16 kHz, mono), además guardamos duraciones
    all_clean_files = []
    for split, ids in splits.items():
        outdir = os.path.join(PROJ, "data", "clean", split)
        os.makedirs(outdir, exist_ok=True)
        for k, i_filtered in enumerate(ids):
            i_original = filtered_indices[i_filtered]
            item = ds[i_original] # (waveform, sample_rate, label, speaker_id, utterance_number)
            wav, sr, label, _, _ = item
            if wav.ndim == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            x = wav.squeeze(0).numpy().astype("float32")
            m = np.max(np.abs(x)) + 1e-9
            x = (x/m).astype("float32")
            filename = f"{label}_{k:04d}.wav"
            sf.write(os.path.join(outdir, filename), x, SR)
            all_clean_files.append((split, filename, len(x)/SR))

    print("Archivos CLEAN guardados y duraciones registradas.")

    # 5.4) Generar NOISE sintético (blanco/rosa/café/babble)
    def gen_white(n): return np.random.randn(n).astype("float32")
    def gen_pink(n):
        rows = 16
        arr = np.zeros(n, dtype=np.float64)
        vals = np.random.randn(rows)
        cnt = np.zeros(rows, dtype=int)
        for i in range(n):
            j = 0
            while True:
                cnt[j] += 1
                if cnt[j] & (1 << j):
                    cnt[j] = 0
                    vals[j] = np.random.randn()
                    j += 1
                    if j == rows: break
                else:
                    break
            arr[i] = vals.sum()
        arr = arr / (np.max(np.abs(arr)) + 1e-9)
        return arr.astype("float32")
    def gen_brown(n):
        w = np.random.randn(n).astype("float32")
        y = np.cumsum(w)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y
    def norm(x):
        m = np.max(np.abs(x)) + 1e-9
        return (x/m).astype("float32")

    def save_noise_set(outdir, seconds_list, babble_samples):
        os.makedirs(outdir, exist_ok=True)
        for k, secs in enumerate(seconds_list):
            n = int(SR*secs)
            choice = random.choice(["white","pink","brown","babble"])
            if choice=="white":
                x = norm(gen_white(n))
            elif choice=="pink":
                x = norm(gen_pink(n))
            elif choice=="brown":
                x = norm(gen_brown(n))
            else: # babble
                mix = np.zeros(n, dtype="float32")
                pick_indices = random.sample(babble_samples, k=min(3, len(babble_samples)))
                for i_original in pick_indices:
                    wav, sr, _, _, _ = ds[i_original]
                    if wav.ndim == 2 and wav.size(0) > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    if sr != SR:
                        wav = torchaudio.functional.resample(wav, sr, SR)
                    s = wav.squeeze(0).numpy().astype("float32")
                    if len(s) < n: s = np.pad(s, (0, n-len(s)))
                    else: s = s[:n]
                    mix += 0.3*s
                x = norm(mix + 0.05*np.random.randn(n).astype("float32"))
            sf.write(os.path.join(outdir, f"noise_{k:04d}.wav"), x, SR)

    split_durations = {
        "train": [d for s, f, d in all_clean_files if s == "train"],
        "val":  [d for s, f, d in all_clean_files if s == "val"],
        "test": [d for s, f, d in all_clean_files if s == "test"],
    }
    babble_samples_idx = filtered_indices

    def get_noise_lengths(clean_durations):
        return [d + random.uniform(0.5, 1.5) for d in clean_durations]

    print("Generando NOISE...")
    save_noise_set(noise_train, get_noise_lengths(split_durations["train"]), babble_samples_idx)
    save_noise_set(noise_val,   get_noise_lengths(split_durations["val"]),   babble_samples_idx)
    save_noise_set(noise_test,  get_noise_lengths(split_durations["test"]),  babble_samples_idx)

    print("---")
    print("Mini-dataset creado: CLEAN=SPEECHCOMMANDS (one/two/three, 16 kHz), NOISE=sintético.")
    print(f"Archivos CLEAN: Train={len(split_durations['train'])}, Val={len(split_durations['val'])}, Test={len(split_durations['test'])}")

# -----------------------------------------------------------------------------
# 6) Métricas y checkpoints
# -----------------------------------------------------------------------------

def batch_metrics(estimate, clean, sr=16000):
    """
    Devuelve promedios de PESQ (wideband 16k) y STOI.
    Si las librerías no están instaladas o clips muy cortos, devuelve NaN.
    """
    try:
        from pesq import pesq as _pesq
        from pystoi.stoi import stoi as _stoi
    except Exception:
        _pesq = None
        _stoi = None

    est = estimate.detach().cpu().numpy()
    ref = clean.detach().cpu().numpy()
    pesq_list, stoi_list = [], []
    for e, r in zip(est, ref):
        L = min(len(e), len(r))
        e = e[:L].astype(np.float32)
        r = r[:L].astype(np.float32)
        e = np.clip(e, -1.0, 1.0); r = np.clip(r, -1.0, 1.0)
        if _pesq is not None:
            try: pesq_list.append(_pesq(sr, r, e, 'wb'))
            except Exception: pass
        if _stoi is not None:
            try: stoi_list.append(_stoi(r, e, sr, extended=False))
            except Exception: pass
    m_pesq = float(np.mean(pesq_list)) if pesq_list else float("nan")
    m_stoi = float(np.mean(stoi_list)) if stoi_list else float("nan")
    return m_pesq, m_stoi

def save_ckpt(model, opt, epoch, path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": config
    }, path)
    return path

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("config", None)
    model = UNetSpec(depth=4).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg

# -----------------------------------------------------------------------------
# 7) Entrenamiento
# -----------------------------------------------------------------------------

def make_loader(split, C, shuffle=True):
    num_workers = 0 if os.name == "nt" else 2  # Windows: evitar multiprocessing
    return DataLoader(
        NoisyPairDataset(
            clean_dir=os.path.join(PROJ, "data", "clean", split),
            noise_dir=os.path.join(PROJ, "data", "noise", split),
            sr=C["sample_rate"],
            clip_seconds=C["clip_seconds"],
            snr_min=C.get("snr",{"min":0})["min"],
            snr_max=C.get("snr",{"max":20})["max"],
        ),
        batch_size=C["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

def train_one_epoch(model, opt, scaler, train_loader, N_FFT, HOP, device):
    model.train()
    running = []
    t0 = time.time()
    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            mag, ph, _ = stft_mag(noisy, N_FFT, HOP)      # [B,1,F,T]
            mask = model(mag)                              # [B,1,F,T]
            est  = istft(mag*mask, ph, N_FFT, HOP)        # [B,T]

            # pérdidas: onda + magnitud
            l_time = F.l1_loss(est, clean)
            est_mag, _, _ = stft_mag(est, N_FFT, HOP)
            clean_mag, _, _ = stft_mag(clean, N_FFT, HOP)
            l_spec = F.l1_loss(est_mag, clean_mag)
            loss = l_time + 0.5*l_spec

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        running.append(loss.item())
    dt = time.time()-t0
    return float(np.mean(running)), dt

@torch.no_grad()
def validate(model, loader, N_FFT, HOP, device, SR):
    model.eval()
    losses, pesqs, stois = [], [], []
    for noisy, clean in loader:
        noisy = noisy.to(device); clean = clean.to(device)
        mag, ph, _ = stft_mag(noisy, N_FFT, HOP)
        mask = model(mag)
        est  = istft(mag*mask, ph, N_FFT, HOP)
        l_time = F.l1_loss(est, clean)
        est_mag, _, _ = stft_mag(est, N_FFT, HOP)
        clean_mag, _, _ = stft_mag(clean, N_FFT, HOP)
        l_spec = F.l1_loss(est_mag, clean_mag)
        loss = l_time + 0.5*l_spec
        losses.append(loss.item())
        p, s = batch_metrics(est, clean, SR)
        if not math.isnan(p): pesqs.append(p)
        if not math.isnan(s): stois.append(s)
    m_loss = float(np.mean(losses)) if losses else float("nan")
    m_pesq = (np.mean(pesqs) if pesqs else float("nan"))
    m_stoi = (np.mean(stois) if stois else float("nan"))
    return m_loss, m_pesq, m_stoi

def run_training(C):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSpec(depth=4).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    N_FFT  = C["n_fft"]
    HOP    = C["hop_length"]
    SR     = C["sample_rate"]

    train_loader = make_loader("train", C, True)
    val_loader   = make_loader("val",   C, False)

    best_val = -1e9
    EPOCHS = C["train"]["epochs"]

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_time = train_one_epoch(model, opt, scaler, train_loader, N_FFT, HOP, device)
        va_loss, va_pesq, va_stoi = validate(model, val_loader, N_FFT, HOP, device, SR)

        print(f"[{ep:03d}] train {tr_loss:.4f} ({tr_time:.1f}s) | val {va_loss:.4f} | PESQ {va_pesq:.3f} | STOI {va_stoi:.3f}")

        # guarda el mejor por (PESQ + STOI); puedes ajustar el criterio
        score = (0 if math.isnan(va_pesq) else va_pesq) + (0 if math.isnan(va_stoi) else va_stoi)
        if score > best_val:
            best_val = score
            ckpt_path = os.path.join(PROJ, "checkpoints", "unet_best.pt")
            ck = save_ckpt(model, opt, ep, ckpt_path, C)
            print("   ✓ checkpoint guardado:", ck)

    return model, C

# -----------------------------------------------------------------------------
# 8) Guardar muestras y denoise por carpeta
# -----------------------------------------------------------------------------

@torch.no_grad()
def save_val_samples(model, C):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_FFT, HOP, SR = C["n_fft"], C["hop_length"], C["sample_rate"]
    val_loader = make_loader("val", C, False)
    model.eval()
    noisy, clean = next(iter(val_loader))
    noisy = noisy.to(device); clean = clean.to(device)
    mag, ph, _ = stft_mag(noisy, N_FFT, HOP)
    mask = model(mag)
    est  = istft(mag*mask, ph, N_FFT, HOP).cpu()

    out_dir = os.path.join(PROJ, "outputs", "samples")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(3, noisy.size(0))):
        save_wav(os.path.join(out_dir, f"noisy_{i}.wav"), noisy[i].cpu().numpy(), SR)
        save_wav(os.path.join(out_dir, f"clean_{i}.wav"), clean[i].cpu().numpy(), SR)
        save_wav(os.path.join(out_dir, f"est_{i}.wav"),   est[i].numpy(), SR)
    print("3 muestras guardadas en outputs/samples/")

@torch.no_grad()
def denoise_folder(in_glob, out_dir, ckpt_path):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt_path, device)
    n_fft = cfg["n_fft"] if cfg else 512
    hop   = cfg["hop_length"] if cfg else 128
    sr    = cfg["sample_rate"] if cfg else 16000

    paths = glob.glob(in_glob)
    if not paths:
        print(f"No se encontraron archivos con patrón: {in_glob}")
        return

    for path in paths:
        x = load_wav(path, sr)
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        mag, ph, _ = stft_mag(t, n_fft, hop)
        mask = model(mag)
        y = istft(mag*mask, ph, n_fft, hop).squeeze(0).cpu().numpy()
        save_wav(os.path.join(out_dir, os.path.basename(path)), y, sr)
    print(f"Listo: archivos guardados en {out_dir}")

# -----------------------------------------------------------------------------
# 9) Config por defecto
# -----------------------------------------------------------------------------

DEFAULT_CFG = {
  "sample_rate": 16000,
  "n_fft": 512,
  "hop_length": 128,
  "clip_seconds": 3,
  "batch_size": 16,
  "lr": 1e-3,
  "snr": {"min": 0, "max": 20},
  "train": {
    "clean_dir": "data/clean",
    "noise_dir": "data/noise",
    "epochs": 10
  },
  "paths": {
    "ckpt_dir": "checkpoints",
    "out_dir": "outputs"
  }
}

def ensure_config(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CFG, f, sort_keys=False)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# 10) Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(1337)
    cfg_path = os.path.join(PROJ, "configs", "base.yaml")
    C = ensure_config(cfg_path)

    # Crear/descargar dataset si hace falta
    prepare_mini_dataset(C, force=False)

    # Entrenar
    model, C = run_training(C)

    # Guardar 3 muestras de validación
    save_val_samples(model, C)

    # Inferencia por carpeta (si hay WAVs en ./inputs)
    in_glob = os.path.join(PROJ, "inputs", "*.wav")
    out_dir = os.path.join(PROJ, "outputs", "denoised")
    ckpt    = os.path.join(PROJ, "checkpoints", "unet_best.pt")
    if os.path.exists(ckpt):
        denoise_folder(in_glob, out_dir, ckpt)
    else:
        print("No hay checkpoint para inferencia aún:", ckpt)
