from __future__ import annotations
from pathlib import Path


def discover_BRAS(path_to_BRAS: Path) -> list[Path]:
    """
    Discover all BRAS files in a directory.

    Args:
        path_to_BRAS (Path):
            Path to the directory containing BRAS files.
    Returns:
        list[Path]:
            List of paths to RIR files.
    """
    all_files = list(path_to_BRAS.glob("**/RIRs/wav/*.wav"))
    all_files.sort()
    return all_files


def discover_librispeech(path_to_librispeech: Path) -> list[Path]:
    """
    Discover all LibriSpeech files in a directory.

    Args:
        path_to_librispeech (Path):
            Path to the directory containing LibriSpeech files.
    Returns:
        list[Path]:
            List of paths to source files.
    """
    all_files = list(path_to_librispeech.glob("**/*.flac"))
    all_files.sort()
    return all_files


def discover_Whamr(path_to_Whamr: Path) -> list[Path]:
    """
    Discover all WHAMR files in a directory.

    Args:
        path_to_Whamr (Path):
            Path to the directory containing WHAMR files.
    Returns:
        list[Path]:
            List of paths to noise files.
    """
    all_files = list(path_to_Whamr.glob("**/*.wav"))
    all_files.sort()
    return all_files
