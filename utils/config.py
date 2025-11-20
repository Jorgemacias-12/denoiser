from __future__ import annotations
import os
from typing import Dict, Optional
from colorama import Fore, Style
from pathlib import Path


class Config:
    """
    Singleton that stores the config data loaded from an .env file
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, str]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = {}

        return cls._instance

    def load(self, file_path: str) -> None:
        """
        Loads the keys/values of the .env file to an internal dictionary file
        """

        if not os.path.exists(file_path):
            print(
                f"{Style.BRIGHT}{Fore.RED}[ERROR] Archivo no encontrado {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                self._config[key.strip()] = value.strip()

    def get(self, key: str, default=None) -> Optional[str]:
        return self._config.get(key, default)

    def all(self) -> Dict[str, str]:
        return dict(self._config)


config = Config()

BASE_DIR = Path(__file__).resolve().parent.parent  # carpeta raíz donde está __main__.py
ENV_PATH = BASE_DIR / ".env"

config.load(str(ENV_PATH))
