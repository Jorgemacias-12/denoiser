from pathlib import Path
from colorama import Fore, Style, init
from tqdm import tqdm
import soundfile as sf
import numpy as np
import random
import os

# Intentar importar torchaudio con fallback
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError as e:
    print(f"{Fore.YELLOW}Advertencia: torchaudio no disponible. Usando métodos alternativos.")
    print(f"Error: {e}")
    TORCHAUDIO_AVAILABLE = False

# Inicializar colorama
init(autoreset=True)

# Definir el directorio raíz del proyecto
PROJECT_ROOT = Path("./")


def resample_audio(audio_data, original_sr, target_sr):
    """
    Resample audio usando métodos básicos si torchaudio no está disponible
    """
    if TORCHAUDIO_AVAILABLE:
        return torchaudio.functional.resample(audio_data, original_sr, target_sr)
    else:
        # Resample simple usando scipy (fallback)
        try:
            from scipy import signal
            ratio = target_sr / original_sr
            new_length = int(len(audio_data) * ratio)
            return signal.resample(audio_data, new_length)
        except ImportError:
            # Si scipy no está disponible, devolver el audio original con advertencia
            print(
                f"{Fore.YELLOW}Advertencia: No se pudo resamplear. Usando audio original.")
            return audio_data


def load_audio_file(file_path, target_sr):
    """
    Cargar archivo de audio usando soundfile como fallback
    """
    try:
        audio_data, sr = sf.read(str(file_path))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convertir a mono
        if sr != target_sr:
            audio_data = resample_audio(audio_data, sr, target_sr)
        return audio_data
    except Exception as e:
        print(f"{Fore.RED}Error cargando {file_path}: {e}")
        return None


def prepare_mini_dataset(force_recreate=False):
    """
    Crea datasets CLEAN y NOISE sintético usando métodos compatibles.
    """
    SAMPLE_RATE = 16000

    # Definir rutas de directorios
    clean_data_dirs = {
        "train": PROJECT_ROOT / "data" / "clean" / "train",
        "val": PROJECT_ROOT / "data" / "clean" / "val",
        "test": PROJECT_ROOT / "data" / "clean" / "test"
    }

    noise_data_dirs = {
        "train": PROJECT_ROOT / "data" / "noise" / "train",
        "val": PROJECT_ROOT / "data" / "noise" / "val",
        "test": PROJECT_ROOT / "data" / "noise" / "test"
    }

    # Crear todos los directorios necesarios
    all_directories = list(clean_data_dirs.values()) + \
        list(noise_data_dirs.values())
    for directory_path in all_directories:
        directory_path.mkdir(parents=True, exist_ok=True)

    # Verificar si ya existen datos
    has_existing_clean_data = any(clean_data_dirs["train"].iterdir())
    has_existing_noise_data = any(noise_data_dirs["train"].iterdir())

    if has_existing_clean_data and has_existing_noise_data and not force_recreate:
        print(f"{Fore.YELLOW}Dataset ya existe. Saltando creación.")
        return

    # Si torchaudio no está disponible, usar dataset preexistente o generar datos sintéticos
    if not TORCHAUDIO_AVAILABLE:
        print(
            f"{Fore.YELLOW}Torchaudio no disponible. Generando dataset sintético básico.")
        generate_synthetic_dataset(
            SAMPLE_RATE, clean_data_dirs, noise_data_dirs)
        return

    # Descargar y cargar dataset SpeechCommands
    speech_commands_dataset_path = PROJECT_ROOT / "speechcommands"
    speech_commands_dataset_path.mkdir(parents=True, exist_ok=True)

    print(f"{Fore.CYAN}Descargando/leyendo SPEECHCOMMANDS...")
    try:
        speech_commands_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(speech_commands_dataset_path),
            download=True
        )
    except Exception as e:
        print(f"{Fore.RED}Error descargando SpeechCommands: {e}")
        print(f"{Fore.YELLOW}Generando dataset sintético alternativo...")
        generate_synthetic_dataset(
            SAMPLE_RATE, clean_data_dirs, noise_data_dirs)
        return

    # Filtrar comandos deseados
    TARGET_COMMANDS = ['one', 'two', 'three']
    filtered_sample_indices = []

    print(f"{Fore.CYAN}Filtrando muestras de comandos: {', '.join(TARGET_COMMANDS)}")
    for sample_index in tqdm(range(len(speech_commands_dataset)), desc="Filtrando comandos"):
        try:
            _, _, command_label, _, _ = speech_commands_dataset[sample_index]
            if command_label in TARGET_COMMANDS:
                filtered_sample_indices.append(sample_index)
        except Exception as error:
            print(f"{Fore.RED}Error procesando muestra {sample_index}: {error}")

    print(f"{Fore.GREEN}Total muestras filtradas: {len(filtered_sample_indices)}")

    if len(filtered_sample_indices) == 0:
        print(
            f"{Fore.YELLOW}No se encontraron muestras. Generando dataset sintético...")
        generate_synthetic_dataset(
            SAMPLE_RATE, clean_data_dirs, noise_data_dirs)
        return

    # Dividir en conjuntos (60/20/20)
    total_filtered_samples = len(filtered_sample_indices)
    shuffled_indices = list(range(total_filtered_samples))
    random.shuffle(shuffled_indices)

    train_split_cutoff = int(0.6 * total_filtered_samples)
    val_split_cutoff = int(0.8 * total_filtered_samples)

    dataset_splits = {
        "train": shuffled_indices[:train_split_cutoff],
        "val": shuffled_indices[train_split_cutoff:val_split_cutoff],
        "test": shuffled_indices[val_split_cutoff:]
    }

    # Guardar archivos CLEAN
    all_clean_files_metadata = []

    print(f"{Fore.CYAN}Guardando archivos CLEAN...")
    for split_name, split_indices in dataset_splits.items():
        output_directory = clean_data_dirs[split_name]

        for progress_index, filtered_index in enumerate(tqdm(split_indices, desc=f"CLEAN {split_name}")):
            try:
                original_dataset_index = filtered_sample_indices[filtered_index]
                dataset_item = speech_commands_dataset[original_dataset_index]

                audio_waveform, original_sample_rate, command_label, _, _ = dataset_item

                # Convertir a mono si es estéreo
                if audio_waveform.ndim == 2 and audio_waveform.size(0) > 1:
                    audio_waveform = audio_waveform.mean(dim=0, keepdim=True)

                # Resamplear
                if original_sample_rate != SAMPLE_RATE:
                    audio_waveform = resample_audio(
                        audio_waveform, original_sample_rate, SAMPLE_RATE)

                # Normalizar audio
                audio_data = audio_waveform.squeeze(
                    0).numpy().astype("float32")
                max_amplitude = np.max(np.abs(audio_data)) + 1e-9
                normalized_audio = (
                    audio_data / max_amplitude).astype("float32")

                # Guardar archivo
                output_filename = f"{command_label}_{progress_index:04d}.wav"
                output_filepath = output_directory / output_filename
                sf.write(str(output_filepath), normalized_audio, SAMPLE_RATE)

                # Registrar metadatos
                audio_duration_seconds = len(normalized_audio) / SAMPLE_RATE
                all_clean_files_metadata.append(
                    (split_name, output_filename, audio_duration_seconds))

            except Exception as e:
                print(f"{Fore.RED}Error procesando archivo {progress_index}: {e}")

    print(f"{Fore.GREEN}Archivos CLEAN guardados: {len(all_clean_files_metadata)}")

    # Generar archivos NOISE
    generate_noise_files(SAMPLE_RATE, noise_data_dirs, all_clean_files_metadata,
                         filtered_sample_indices, speech_commands_dataset)


def generate_synthetic_dataset(sample_rate, clean_dirs, noise_dirs):
    """
    Genera un dataset sintético básico cuando SpeechCommands no está disponible
    """
    print(f"{Fore.CYAN}Generando dataset sintético...")

    TARGET_COMMANDS = ['one', 'two', 'three']
    SAMPLES_PER_SPLIT = 30

    # Generar archivos CLEAN sintéticos
    for split_name, output_dir in clean_dirs.items():
        for i in range(SAMPLES_PER_SPLIT):
            command = random.choice(TARGET_COMMANDS)
            duration = random.uniform(0.8, 1.5)
            samples = int(sample_rate * duration)

            # Generar tono simple para simular voz
            t = np.linspace(0, duration, samples)
            freq = 220 if command == 'one' else 440 if command == 'two' else 660
            audio_data = 0.3 * np.sin(2 * np.pi * freq * t)
            audio_data += 0.1 * np.random.randn(samples)

            # Normalizar
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-9)

            output_filename = f"{command}_{i:04d}.wav"
            output_filepath = output_dir / output_filename
            sf.write(str(output_filepath),
                     audio_data.astype("float32"), sample_rate)

    # Generar archivos NOISE sintéticos
    for split_name, output_dir in noise_dirs.items():
        for i in range(SAMPLES_PER_SPLIT):
            duration = random.uniform(1.0, 2.0)
            generate_and_save_noise(sample_rate, duration, output_dir, i)

    print(f"{Fore.GREEN}Dataset sintético generado: {SAMPLES_PER_SPLIT} muestras por split")


def generate_noise_files(sample_rate, noise_dirs, clean_metadata, babble_indices, dataset=None):
    """
    Genera archivos de ruido
    """
    print(f"{Fore.CYAN}Generando archivos NOISE...")

    # Organizar duraciones por split
    clean_durations_by_split = {
        "train": [duration for split, filename, duration in clean_metadata if split == "train"],
        "val": [duration for split, filename, duration in clean_metadata if split == "val"],
        "test": [duration for split, filename, duration in clean_metadata if split == "test"],
    }

    def get_noise_durations(clean_durations):
        return [duration + random.uniform(0.5, 1.5) for duration in clean_durations]

    for split_name, output_dir in noise_dirs.items():
        durations = get_noise_durations(clean_durations_by_split[split_name])
        for i, duration in enumerate(tqdm(durations, desc=f"NOISE {split_name}")):
            generate_and_save_noise(
                sample_rate, duration, output_dir, i, babble_indices, dataset)


def generate_and_save_noise(sample_rate, duration, output_dir, index, babble_indices=None, dataset=None):
    """
    Genera y guarda un archivo de ruido
    """
    num_samples = int(sample_rate * duration)
    noise_type = random.choice(["white", "pink", "brown", "babble"])

    if noise_type == "white":
        noise_data = generate_white_noise(num_samples)
    elif noise_type == "pink":
        noise_data = generate_pink_noise(num_samples)
    elif noise_type == "brown":
        noise_data = generate_brown_noise(num_samples)
    else:  # babble
        noise_data = generate_babble_noise(
            num_samples, babble_indices, dataset)

    # Normalizar
    noise_data = noise_data / (np.max(np.abs(noise_data)) + 1e-9)

    # Guardar
    output_filename = f"noise_{index:04d}.wav"
    output_filepath = output_dir / output_filename
    sf.write(str(output_filepath), noise_data.astype("float32"), sample_rate)

# Generadores de ruido (mantienen la misma implementación)


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


def generate_babble_noise(num_samples, babble_indices=None, dataset=None):
    """Genera ruido de multitud"""
    babble_mix = np.zeros(num_samples, dtype="float32")

    if dataset is not None and babble_indices:
        selected_indices = random.sample(
            babble_indices, k=min(3, len(babble_indices)))
        for original_index in selected_indices:
            try:
                audio_waveform, original_sr, _, _, _ = dataset[original_index]
                if audio_waveform.ndim == 2 and audio_waveform.size(0) > 1:
                    audio_waveform = audio_waveform.mean(dim=0, keepdim=True)
                if original_sr != SAMPLE_RATE:
                    audio_waveform = resample_audio(
                        audio_waveform, original_sr, SAMPLE_RATE)
                audio_segment = audio_waveform.squeeze(
                    0).numpy().astype("float32")
                if len(audio_segment) < num_samples:
                    audio_segment = np.pad(
                        audio_segment, (0, num_samples - len(audio_segment)))
                else:
                    audio_segment = audio_segment[:num_samples]
                babble_mix += 0.3 * audio_segment
            except:
                continue

    # Si no hay dataset disponible, generar ruido alternativo
    if np.max(np.abs(babble_mix)) < 0.1:
        babble_mix = generate_white_noise(
            num_samples) * 0.5 + generate_brown_noise(num_samples) * 0.5

    return babble_mix + 0.05 * np.random.randn(num_samples).astype("float32")


# Ejemplo de uso
if __name__ == "__main__":
    config = {"sample_rate": 16000}
    prepare_mini_dataset(config, force_recreate=False)
