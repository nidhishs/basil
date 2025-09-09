import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class BasilError(Exception):
    """Custom exception for Basil-related errors."""

    pass


def _prefixed_float32(d: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    """Add prefix to dictionary keys and convert arrays to float32."""
    return {f"{prefix}/{k}": v.astype(np.float32) for k, v in d.items()}


def save_artifacts(
    artifact_dir: Union[str, Path],
    codebooks: Dict[str, np.ndarray],
    preprocess_arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """Save Basil model artifacts to disk.

    Saves codebooks and preprocessing arrays as a compressed NPZ file,
    and metadata as a JSON file in the specified directory.

    Args:
        artifact_dir: Directory path where artifacts will be saved.
        codebooks: Dictionary of codebook arrays keyed by level names.
        preprocess_arrays: Dictionary of preprocessing arrays (mean, components, scales).
        metadata: Dictionary containing model metadata.

    Raises:
        BasilError: If saving fails for any reason.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    all_arrays = {
        **_prefixed_float32(codebooks, "codebooks"),
        **_prefixed_float32(preprocess_arrays, "preprocess"),
    }

    try:
        np.savez_compressed(artifact_dir / "basil.npz", **all_arrays)
    except Exception as e:
        raise BasilError(f"Failed to save basil.npz: {e}")

    try:
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        raise BasilError(f"Failed to save metadata.json: {e}")


def load_artifacts(artifact_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load Basil model artifacts from disk.

    Loads codebooks, preprocessing arrays, and metadata from the specified
    directory containing basil.npz and metadata.json files.

    Args:
        artifact_dir: Directory path containing the artifact files.

    Returns:
        Dictionary with 'codebooks', 'preprocess', and 'metadata' keys.

    Raises:
        BasilError: If directory or files don't exist, or loading fails.
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

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            files = z.files
            codebooks = {
                k.split("/", 1)[1]: z[k] for k in files if k.startswith("codebooks/")
            }
            preprocess = {
                k.split("/", 1)[1]: z[k] for k in files if k.startswith("preprocess/")
            }
    except Exception as e:
        raise BasilError(f"Failed to load NPZ: {e}")

    try:
        with open(json_path) as f:
            metadata = json.load(f)
    except Exception as e:
        raise BasilError(f"Failed to load metadata: {e}")

    return {"codebooks": codebooks, "preprocess": preprocess, "metadata": metadata}


def validate_artifacts(artifacts: Dict[str, Any]) -> None:
    """Validate that artifacts contain all required components.

    Checks that the artifacts dictionary contains the expected structure
    with all required preprocessing arrays, metadata fields, and codebooks.

    Args:
        artifacts: Dictionary returned by load_artifacts().

    Raises:
        BasilError: If any required component is missing or invalid.
    """
    for top in ("codebooks", "preprocess", "metadata"):
        if top not in artifacts:
            raise BasilError(f"Missing '{top}' in artifacts")

    for key in ("mean", "components", "scales"):
        if key not in artifacts["preprocess"]:
            raise BasilError(f"Missing preprocessing array: {key}")

    for key in ("levels", "k_per_level", "dim_in", "dim_pca"):
        if key not in artifacts["metadata"]:
            raise BasilError(f"Missing metadata field: {key}")

    levels = artifacts["metadata"]["levels"]
    expected = {f"C{i+1}" for i in range(levels)}
    actual = set(artifacts["codebooks"])
    if expected != actual:
        raise BasilError(f"Codebook mismatch. Expected: {expected}, Got: {actual}")
