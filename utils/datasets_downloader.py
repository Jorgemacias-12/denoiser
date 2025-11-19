from utils.dataset_downloader import DatasetDownloader
from utils.config import config
from colorama import Fore


def download_all_datasets() -> dict:
    """
    Descarga todos los datasets definidos en dataset_list.

    Retorna:
        dict con formato:
        {
            "clean": "/ruta/al/dataset/clean",
            "noise": "/ruta/al/dataset/noise",
            ...
        }
    """

    dataset_list = config.get("dataset_list")
    if not dataset_list:
        print(
            Fore.RED + "[ERROR] dataset_list no está definido en .env")
        return {}

    names = [name.strip() for name in dataset_list.split(",")]
    output_paths = {}

    for ds in names:
        print(Fore.CYAN +
              f"\n[INFO] Procesando dataset: {ds}")

        url = config.get(f"dataset_{ds}_url")
        name = config.get(f"dataset_{ds}_name", ds)
        root = config.get(f"dataset_{ds}_dir", f"./data/raw/{ds}")

        if not url:
            print(
                Fore.RED + f"[ERROR] Falta dataset_{ds}_url en .env")
            continue

        # Instancia del downloader base
        downloader = DatasetDownloader(root=root)
        downloader.url = url
        downloader.dataset_name = name

        # Ejecutar pipeline
        downloader.run()

        # Guardar path final
        output_paths[ds] = downloader.extract_dir

    return output_paths
