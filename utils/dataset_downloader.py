# utils/dataset_downloader.py
from __future__ import annotations

import os
import tarfile
import zipfile
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from utils.config import config
from colorama import Fore, Style


# ------------------------------------------------------------
# HELPER: Safe extraction to prevent path traversal attacks
# ------------------------------------------------------------
def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract a tarfile, ensuring no file escapes the target directory.

    This prevents exploits via filenames containing "../" or absolute paths.
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_path = os.path.abspath(member_path)

        if not abs_path.startswith(os.path.abspath(path) + os.sep):
            raise Exception("Path traversal detected in tar file")

    tar.extractall(path)


# ------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------
class DatasetDownloader:
    """
    Handles downloading and extracting large datasets safely.

    - Automatically names files based on the dataset URL.
    - Automatically detects the extracted directory (e.g., 'LibriSpeech').
    - Avoids re-downloading or re-extracting if already present.
    """

    SUPPORTED = (".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip", ".gz")

    def __init__(self, root: str | None = None):
        """
        Initialize the downloader using config or defaults.

        Args:
            root: Optional explicit root path for dataset files.
                  If None, uses 'dataset_dir' from config or "./data/raw".
        """
        self.root = root or config.get("dataset_dir") or "./data/raw"
        self.url = config.get("dataset_url")
        self.dataset_name = config.get("dataset_name") or "dataset"

        os.makedirs(self.root, exist_ok=True)

        # Determine filename from URL
        if self.url:
            parsed = urlparse(self.url)
            base = os.path.basename(parsed.path)
            self.filename = base if base else f"{self.dataset_name}.download"
        else:
            self.filename = f"{self.dataset_name}.download"

        self.target_path = os.path.join(self.root, self.filename)
        self.extract_dir: str | None = None

    # ------------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------------
    def download(self, force: bool = False) -> str:
        """
        Downloads the dataset file unless it already exists.

        Args:
            force: If True, re-downloads the file even if it exists.

        Returns:
            Path to the downloaded file.
        """
        if not self.url:
            raise ValueError(
                f"{Fore.RED}dataset_url está vacío. Defínelo en .env{Style.RESET_ALL}"
            )

        # Skip download if file already exists
        if os.path.exists(self.target_path) and not force:
            print(
                Fore.YELLOW
                + f"[INFO] Archivo ya existe en {self.target_path}, saltando descarga."
                + Style.RESET_ALL
            )
            return self.target_path

        print(
            Fore.CYAN + f"[INFO] Descargando dataset: {self.dataset_name}" + Style.RESET_ALL)
        print(Fore.CYAN + f"[INFO] URL: {self.url}" + Style.RESET_ALL)

        with requests.get(self.url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 64 * 1024  # 64 KB

            with open(self.target_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    written = f.write(chunk)
                    pbar.update(written)

        print(Fore.GREEN + "[INFO] Download complete." + Style.RESET_ALL)
        return self.target_path

    # ------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------
    def extract(self) -> None:
        """
        Extracts the dataset safely depending on the file type.

        Supports: .tar.gz .tgz .tar.bz2 .tar .zip .gz
        """

        print(Fore.CYAN + "[INFO] Extracting dataset..." + Style.RESET_ALL)
        path = self.target_path
        lower = path.lower()

        try:
            if lower.endswith((".tar.gz", ".tgz", ".tar")):
                with tarfile.open(path, "r:*") as tar:
                    _safe_extract_tar(tar, self.root)

            elif lower.endswith(".tar.bz2"):
                with tarfile.open(path, "r:bz2") as tar:
                    _safe_extract_tar(tar, self.root)

            elif lower.endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        member_path = os.path.join(self.root, member)
                        abs_path = os.path.abspath(member_path)

                        if not abs_path.startswith(os.path.abspath(self.root) + os.sep):
                            raise Exception(
                                "Path traversal detected in zip file")

                    zip_ref.extractall(self.root)

            elif lower.endswith(".gz") and not lower.endswith(".tar.gz"):
                import gzip
                output = path[:-3]

                with gzip.open(path, "rb") as gz_in, open(output, "wb") as out_f:
                    out_f.write(gz_in.read())

            else:
                raise ValueError(f"Unknown archive format: {path}")

        except Exception as e:
            print(
                Fore.RED + f"[ERROR] Extracción falló: {e}" + Style.RESET_ALL)
            raise

        print(Fore.GREEN + "[INFO] Extraction complete." + Style.RESET_ALL)
        self.extract_dir = self._detect_extracted_dir()

    # ------------------------------------------------------------
    # DETECT EXTRACTED DIRECTORY
    # ------------------------------------------------------------
    def _detect_extracted_dir(self) -> str | None:
        """
        Detects the folder created by extraction.

        - LibriSpeech always extracts to "LibriSpeech/"
        - If not found, returns the first non-empty folder.

        Returns:
            Full path to extracted dataset directory or None if not found.
        """
        candidates = [
            name for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]

        # LibriSpeech special case
        if "LibriSpeech" in candidates:
            return os.path.join(self.root, "LibriSpeech")

        # Fallback: first non-empty directory
        for name in candidates:
            full = os.path.join(self.root, name)
            if os.listdir(full):
                return full

        return None

    # ------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------
    def run(self, force_download: bool = False) -> None:
        """
        Executes the pipeline:
        1) Download file if needed
        2) Extract it if needed
        3) Detect dataset directory

        Skips steps intelligently if dataset already exists.
        """

        # If extracted dataset was already present → skip everything
        existing_dir = self._detect_extracted_dir()

        if existing_dir:
            print(
                Fore.YELLOW
                + f"[INFO] Dataset ya existe en: {existing_dir}, no se descargará."
                + Style.RESET_ALL
            )
            self.extract_dir = existing_dir
            return

        # Otherwise, full pipeline
        file_path = self.download(force=force_download)
        self.extract()

        print(Fore.CYAN +
              f"[INFO] Dataset ready at: {self.extract_dir}" + Style.RESET_ALL)
