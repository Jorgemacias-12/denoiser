import os
import stat
import time
from colorama import Fore


def make_file_writable(path: str):
    """Set read/write and remove read-only flag."""
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.chmod(path, 0o666)
    except:
        pass


def remove_zone_identifier(path: str):
    """Remove Windows block flag from downloaded/extracted files."""
    ads = path + ":Zone.Identifier"
    if os.path.exists(ads):
        try:
            os.remove(ads)
        except:
            pass


def ensure_file_unlocked(path: str, retries: int = 10, delay: float = 0.3):
    """
    Ensures file is not locked/block by antivirus or Windows.
    Retries multiple times before giving up.
    """

    for _ in range(retries):
        try:
            # try opening the file for read/write
            with open(path, "rb+"):
                return True  # unlocked
        except PermissionError:
            time.sleep(delay)

    print(Fore.YELLOW +
          f"[WARN] Archivo sigue bloqueado: {path}")
