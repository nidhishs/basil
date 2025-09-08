"""
I/O utilities for BASIL artifacts.

Handles loading and saving of NPZ files and JSON metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class BasilError(Exception):
    """Custom exception for BASIL errors."""

    pass


def save_artifacts(
    artifact_dir: Union[str, Path],
    codebooks: Dict[str, np.ndarray],
    preprocess_arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """
    Save BASIL artifacts to disk.

    Creates basil.npz with codebooks and preprocessing arrays,
    and metadata.json with configuration.

    Args:
        artifact_dir: Directory to save artifacts.
        codebooks: Dictionary of codebook arrays (C1, C2, ...).
        preprocess_arrays: Dictionary of preprocessing arrays.
        metadata: Metadata dictionary.

    Raises:
        BasilError: If saving fails.
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Combine all arrays for NPZ with prefixed keys
    all_arrays = {
        **{f"codebooks/{k}": v.astype(np.float32) for k, v in codebooks.items()},
        **{
            f"preprocess/{k}": v.astype(np.float32)
            for k, v in preprocess_arrays.items()
        },
    }

    # Helper for atomic save
    def atomic_save(path: Path, save_func) -> None:
        # Insert .tmp before the file extension
        temp_path = path.parent / f"{path.stem}.tmp{path.suffix}"
        try:
            save_func(temp_path)
            temp_path.replace(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise BasilError(f"Failed to save {path.name}: {e}")

    # Save NPZ and JSON atomically
    atomic_save(
        artifact_dir / "basil.npz", lambda p: np.savez_compressed(p, **all_arrays)
    )
    atomic_save(
        artifact_dir / "metadata.json",
        lambda p: json.dump(metadata, open(p, "w"), indent=2),
    )


def load_artifacts(artifact_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load BASIL artifacts from disk.

    Args:
        artifact_dir: Directory containing artifacts.

    Returns:
        Dictionary with 'codebooks', 'preprocess', and 'metadata'.

    Raises:
        BasilError: If loading fails or files are missing.
    """

    artifact_dir = Path(artifact_dir)

    if not artifact_dir.exists():
        raise BasilError(f"Artifact directory not found: {artifact_dir}")

    npz_path = artifact_dir / "basil.npz"
    json_path = artifact_dir / "metadata.json"

    if not npz_path.exists():
        raise BasilError(f"NPZ file not found: {npz_path}")
    if not json_path.exists():
        raise BasilError(f"Metadata file not found: {json_path}")

    # Load NPZ and metadata
    try:
        npz_data = np.load(npz_path)
    except Exception as e:
        raise BasilError(f"Failed to load NPZ: {e}")

    try:
        with open(json_path) as f:
            metadata = json.load(f)
    except Exception as e:
        raise BasilError(f"Failed to load metadata: {e}")

    # Parse NPZ contents into dictionaries
    codebooks = {
        k.split("/", 1)[1]: npz_data[k]
        for k in npz_data.files
        if k.startswith("codebooks/")
    }
    preprocess = {
        k.split("/", 1)[1]: npz_data[k]
        for k in npz_data.files
        if k.startswith("preprocess/")
    }

    return {"codebooks": codebooks, "preprocess": preprocess, "metadata": metadata}


def validate_artifacts(artifacts: Dict[str, Any]) -> None:
    """
    Validate loaded artifacts have required structure.

    Args:
        artifacts: Loaded artifacts dictionary.

    Raises:
        BasilError: If validation fails.
    """

    if "codebooks" not in artifacts:
        raise BasilError("Missing 'codebooks' in artifacts")
    if "preprocess" not in artifacts:
        raise BasilError("Missing 'preprocess' in artifacts")
    if "metadata" not in artifacts:
        raise BasilError("Missing 'metadata' in artifacts")

    # Check required preprocessing arrays
    preprocess = artifacts["preprocess"]
    required_preprocess = ["mean", "components", "scales"]
    for key in required_preprocess:
        if key not in preprocess:
            raise BasilError(f"Missing preprocessing array: {key}")

    # Check metadata fields
    metadata = artifacts["metadata"]
    required_metadata = ["levels", "k_per_level", "dim_in", "dim_pca"]
    for key in required_metadata:
        if key not in metadata:
            raise BasilError(f"Missing metadata field: {key}")

    # Check codebooks match metadata
    levels = metadata["levels"]
    expected_codebooks = {f"C{i+1}" for i in range(levels)}
    actual_codebooks = set(artifacts["codebooks"].keys())

    if expected_codebooks != actual_codebooks:
        raise BasilError(
            f"Codebook mismatch. Expected: {expected_codebooks}, "
            f"Got: {actual_codebooks}"
        )
