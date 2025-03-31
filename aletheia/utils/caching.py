import os
from pathlib import Path


_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".aletheia_cache")
aletheia_CACHEDIR = os.environ.get("aletheia_CACHEDIR") or _DEFAULT_CACHE_DIR


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the aletheia cache directory."""
    subdir = os.path.join(aletheia_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir
