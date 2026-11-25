"""
Initialization for configuration package.
Sets environment variables such as MPLCONFIGDIR before matplotlib is loaded.
"""
import os
from pathlib import Path

# Ensure Matplotlib uses local writable directory
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".mpl_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_DIR)
