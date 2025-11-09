from pathlib import Path
from colorama import Fore, init
from tqdm import tqdm
import soundfile as sf
import numpy as np
import random

# Inicializar colorama
init(autoreset=True)

PROJECT_ROOT = Path("./")
SAMPLE_RATE = 16000


def prepare_mini_dataset(force_recreate=False):
    """
    Versión completamente independiente - no usa torchaudio ni torchcodec
    """

    noise_dirs = {
        "train": PROJECT_ROOT / "data" / "noise" / "train",
        "val": PROJECT_ROOT / "data" / "noise" / "val",
        "test": PROJECT_ROOT / "data" / "noise" / "test"
    }

    print(f"{Fore.CYAN}Generando dataset completamente sintético a {SAMPLE_RATE} Hz...")

    # Solo generar datos sintéticos - sin descargar SpeechCommands
    generate_complete_synthetic_dataset(noise_dirs)

    print(f"{Fore.GREEN}✅ Dataset sintético creado exitosamente!")

    print(
        f"{Fore.GREEN}📍 NOISE: {len(list(noise_dirs['train'].iterdir()))} archivos por split")


def generate_complete_synthetic_dataset(noise_dirs):
    """Genera un dataset completamente sintético"""

    # 1. Generar datos NOISE (ruidos ambientales sintéticos)
    print(f"{Fore.CYAN}🔊 Generando ruidos ambientales sintéticos...")
    generate_synthetic_environmental_noise(noise_dirs)


def generate_synthetic_speech_commands(clean_dirs):
    """Genera comandos de voz 'one', 'two', 'three' sintéticos"""
    SAMPLES_PER_SPLIT = {
        "train": 60,  # 20 de cada comando
        "val": 20,    # ~7 de cada comando
        "test": 20    # ~7 de cada comando
    }

    TARGET_WORDS = ['one', 'two', 'three']

    for split_name, output_dir in clean_dirs.items():
        samples_per_word = SAMPLES_PER_SPLIT[split_name] // len(TARGET_WORDS)

        for word in TARGET_WORDS:
            for i in tqdm(range(samples_per_word), desc=f"🗣️  {word} {split_name}"):
                audio_data = generate_speech_command_audio(word)
                filename = f"{word}_{split_name}_{i:04d}.wav"
                output_path = output_dir / filename
                sf.write(str(output_path), audio_data, SAMPLE_RATE)


def generate_speech_command_audio(word):
    """Genera audio sintético para un comando de voz específico"""

    # Parámetros base según la palabra
    if word == 'one':
        base_freq = 180
        duration = random.uniform(0.7, 1.1)
        formants = [500, 1500, 2500]  # Formantes vocálicos para "one"
    elif word == 'two':
        base_freq = 220
        duration = random.uniform(0.6, 0.9)
        formants = [600, 1700, 2600]  # Formantes para "two"
    else:  # 'three'
        base_freq = 260
        duration = random.uniform(0.9, 1.3)
        formants = [400, 1600, 2700]  # Formantes para "three"

    return generate_realistic_speech(duration, base_freq, formants, word)


def generate_realistic_speech(duration, pitch, formants, word):
    """Genera audio de voz más realista usando síntesis por formantes"""
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples)

    # Señal principal con variación de pitch (vibrato)
    pitch_variation = pitch * \
        (1 + 0.02 * np.sin(2 * np.pi * 5 * t))  # Vibrato a 5Hz
    fundamental = np.sin(2 * np.pi * pitch_variation * t)

    # Formantes (resonancias del tracto vocal)
    speech_signal = np.zeros(num_samples)

    # Agregar formantes
    for i, formant_freq in enumerate(formants):
        # Formantes más altos tienen menos energía
        formant_gain = 0.3 / (i + 1)
        formant_bw = formant_freq * 0.05  # Ancho de banda del formante

        # Filtro de formante simplificado
        formant_signal = formant_gain * np.sin(2 * np.pi * formant_freq * t)
        speech_signal += formant_signal

    # Combinar fundamental y formantes
    audio = 0.7 * fundamental + 0.3 * speech_signal

    # Envolvente temporal específica para cada palabra
    envelope = generate_word_envelope(t, duration, word)
    audio *= envelope

    # Ruido de aspiración
    aspiration = 0.05 * np.random.randn(num_samples) * envelope
    audio += aspiration

    # Efecto de reverberación simple
    audio = add_simple_reverb(audio, SAMPLE_RATE)

    # Normalizar
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio.astype("float32")


def generate_word_envelope(t, duration, word):
    """Genera envolventes temporales específicas para cada palabra"""
    if word == 'one':
        # "one" - ataque rápido, decaimiento medio
        attack = np.minimum(t / 0.1, 1.0)
        decay = np.exp(-2 * (t - duration/2) ** 2 / (duration/3) ** 2)
        return attack * decay

    elif word == 'two':
        # "two" - dos sílabas
        syllable1 = np.exp(-3 * (t - duration/3) ** 2 / (duration/5) ** 2)
        syllable2 = np.exp(-3 * (t - 2*duration/3) ** 2 / (duration/5) ** 2)
        return syllable1 + 0.8 * syllable2

    else:  # 'three'
        # "three" - tres sílabas
        syl1 = np.exp(-4 * (t - duration/4) ** 2 / (duration/6) ** 2)
        syl2 = np.exp(-4 * (t - duration/2) ** 2 / (duration/6) ** 2)
        syl3 = np.exp(-4 * (t - 3*duration/4) ** 2 / (duration/6) ** 2)
        return syl1 + 0.7 * syl2 + 0.5 * syl3


def add_simple_reverb(audio, sample_rate, decay_time=0.3):
    """Añade reverberación simple"""
    reverb_samples = int(sample_rate * decay_time)
    reverb = np.exp(-np.linspace(0, 5, reverb_samples))

    # Aplicar reverberación (convolución simple)
    reverberated = np.convolve(audio, reverb, mode='same')

    # Mezclar con señal original
    mix = 0.7 * audio + 0.3 * reverberated
    return mix


def generate_synthetic_environmental_noise(noise_dirs):
    """Genera ruidos ambientales sintéticos realistas"""
    SAMPLES_PER_SPLIT = {
        "train": 60,
        "val": 20,
        "test": 20
    }

    NOISE_TYPES = ['white', 'pink', 'brown', 'urban', 'nature', 'electronic']

    for split_name, output_dir in noise_dirs.items():
        for i in tqdm(range(SAMPLES_PER_SPLIT[split_name]), desc=f"🔊 NOISE {split_name}"):
            duration = random.uniform(1.5, 3.0)
            noise_type = random.choice(NOISE_TYPES)
            noise_data = generate_environmental_noise(duration, noise_type)

            filename = f"noise_{noise_type}_{split_name}_{i:04d}.wav"
            output_path = output_dir / filename
            sf.write(str(output_path), noise_data, SAMPLE_RATE)


def generate_environmental_noise(duration, noise_type):
    """Genera diferentes tipos de ruido ambiental"""
    num_samples = int(SAMPLE_RATE * duration)

    if noise_type == 'white':
        return generate_white_noise(num_samples)
    elif noise_type == 'pink':
        return generate_pink_noise(num_samples)
    elif noise_type == 'brown':
        return generate_brown_noise(num_samples)
    elif noise_type == 'urban':
        return generate_urban_noise(num_samples)
    elif noise_type == 'nature':
        return generate_nature_noise(num_samples)
    else:  # 'electronic'
        return generate_electronic_noise(num_samples)


def generate_urban_noise(num_samples):
    """Genera ruido urbano (tráfico, multitudes)"""
    t = np.linspace(0, num_samples/SAMPLE_RATE, num_samples)

    # Base de ruido de tráfico
    traffic = 0.4 * generate_brown_noise(num_samples)

    # Eventos aleatorios (bocinas, etc.)
    events = np.zeros(num_samples)
    num_events = random.randint(2, 5)
    for _ in range(num_events):
        event_pos = random.randint(0, num_samples-1)
        event_duration = random.randint(1000, 5000)  # 0.06-0.3 segundos
        event_end = min(event_pos + event_duration, num_samples-1)
        events[event_pos:event_end] += 0.3 * \
            np.random.randn(event_end - event_pos)

    return (traffic + events) / (np.max(np.abs(traffic + events)) + 1e-9)


def generate_nature_noise(num_samples):
    """Genera ruido de naturaleza (viento, pájaros)"""
    t = np.linspace(0, num_samples/SAMPLE_RATE, num_samples)

    # Viento (ruido rosa con modulación)
    wind = 0.6 * generate_pink_noise(num_samples)
    wind_modulation = 0.5 + 0.5 * \
        np.sin(2 * np.pi * 0.2 * t)  # Modulación lenta
    wind *= wind_modulation

    # Pájaros (tonos agudos aleatorios)
    birds = np.zeros(num_samples)
    num_bird_calls = random.randint(3, 8)
    for _ in range(num_bird_calls):
        call_pos = random.randint(0, num_samples-1)
        call_freq = random.uniform(2000, 5000)  # Pájaros agudos
        call_duration = random.randint(100, 300)  # 0.006-0.02 segundos
        call_end = min(call_pos + call_duration, num_samples-1)
        call_t = np.linspace(0, call_duration/SAMPLE_RATE, call_end - call_pos)
        birds[call_pos:call_end] += 0.2 * \
            np.sin(2 * np.pi * call_freq * call_t)

    return (wind + birds) / (np.max(np.abs(wind + birds)) + 1e-9)


def generate_electronic_noise(num_samples):
    """Genera ruido electrónico (estática, interferencia)"""
    # Ruido blanco con picos agudos
    base = generate_white_noise(num_samples)

    # Picos de interferencia
    t = np.linspace(0, num_samples/SAMPLE_RATE, num_samples)
    interference = np.zeros(num_samples)

    num_interference = random.randint(5, 15)
    for _ in range(num_interference):
        freq = random.uniform(1000, 8000)
        pos = random.randint(0, num_samples-1)
        duration = random.randint(50, 200)
        end = min(pos + duration, num_samples-1)
        interference[pos:end] += 0.4 * np.sin(2 * np.pi * freq * t[pos:end])

    return (0.7 * base + 0.3 * interference) / (np.max(np.abs(0.7 * base + 0.3 * interference)) + 1e-9)

# Generadores de ruido básicos (mantenidos igual)


def generate_white_noise(num_samples):
    return np.random.randn(num_samples).astype("float32")


def generate_pink_noise(num_samples):
    num_rows = 16
    pink_noise_buffer = np.zeros(num_samples, dtype=np.float64)
    row_values = np.random.randn(num_rows)
    row_counters = np.zeros(num_rows, dtype=int)

    for i in range(num_samples):
        current_row = 0
        while True:
            row_counters[current_row] += 1
            if row_counters[current_row] & (1 << current_row):
                row_counters[current_row] = 0
                row_values[current_row] = np.random.randn()
                current_row += 1
                if current_row == num_rows:
                    break
            else:
                break
        pink_noise_buffer[i] = row_values.sum()

    pink_noise_buffer = pink_noise_buffer / \
        (np.max(np.abs(pink_noise_buffer)) + 1e-9)
    return pink_noise_buffer.astype("float32")


def generate_brown_noise(num_samples):
    white_noise = np.random.randn(num_samples).astype("float32")
    brownian_noise = np.cumsum(white_noise)
    brownian_noise = brownian_noise / (np.max(np.abs(brownian_noise)) + 1e-9)
    return brownian_noise


# Ejecutar
if __name__ == "__main__":
    prepare_mini_dataset_independent(force_recreate=True)
