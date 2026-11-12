from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np


def hash_embedding_inputs(
    positive: Iterable[str],
    negative: Iterable[str],
) -> str:
    """Return a stable digest of the ordered texts and their labels."""
    digest = hashlib.sha256()
    for label, texts in (("positive", positive), ("negative", negative)):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        for text in texts:
            payload = str(text).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class EmbeddingCheckpoint:
    """Durable, append-only storage for per-sample, per-layer embeddings."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        *,
        identity: Mapping[str, Any],
        n_pos: int,
        n_neg: int,
        layers: int,
        dim: int,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.directory / "metadata.json"
        self.progress_path = self.directory / "progress.json"
        self.data_path = self.directory / "embeddings.float32"
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.layers = int(layers)
        self.dim = int(dim)
        self.total = self.n_pos + self.n_neg
        if min(self.n_pos, self.n_neg, self.layers, self.dim) < 0:
            raise ValueError("embedding checkpoint dimensions must be non-negative")
        if self.total == 0 or self.layers == 0 or self.dim == 0:
            raise ValueError("embedding checkpoint cannot be empty")

        expected_meta = {
            "schema_version": self.SCHEMA_VERSION,
            "identity": dict(identity),
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "layers": self.layers,
            "dim": self.dim,
            "dtype": "float32",
        }
        expected_bytes = self.total * self.layers * self.dim * 4

        if self.meta_path.is_file():
            with self.meta_path.open("r", encoding="utf-8") as handle:
                actual_meta = json.load(handle)
            if actual_meta != expected_meta:
                raise ValueError(
                    f"checkpoint identity mismatch at {self.directory}"
                )
            if not self.data_path.is_file():
                raise ValueError(
                    f"checkpoint data is missing at {self.data_path}"
                )
            if self.data_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"checkpoint data size mismatch at {self.data_path}"
                )
        else:
            for stale_path in (
                self.data_path,
                self.progress_path,
                self.meta_path.with_suffix(".json.tmp"),
                self.progress_path.with_suffix(".json.tmp"),
            ):
                stale_path.unlink(missing_ok=True)
            initial = np.memmap(
                self.data_path,
                mode="w+",
                dtype=np.float32,
                shape=(self.total, self.layers, self.dim),
            )
            initial.flush()
            del initial
            with self.data_path.open("rb") as handle:
                os.fsync(handle.fileno())
            _atomic_json(self.meta_path, expected_meta)
            _atomic_json(self.progress_path, {"completed": 0})

        if self.progress_path.is_file():
            with self.progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
            self.completed = int(progress["completed"])
        else:
            self.completed = 0
            _atomic_json(self.progress_path, {"completed": 0})
        if not 0 <= self.completed <= self.total:
            raise ValueError(
                f"invalid checkpoint progress {self.completed}/{self.total}"
            )

        self._data = np.memmap(
            self.data_path,
            mode="r+",
            dtype=np.float32,
            shape=(self.total, self.layers, self.dim),
        )

    def append(self, index: int, layer_vectors: np.ndarray) -> None:
        if int(index) != self.completed:
            raise ValueError(
                f"checkpoint append index {index} does not match "
                f"completed={self.completed}"
            )
        values = np.asarray(layer_vectors, dtype=np.float32)
        expected_shape = (self.layers, self.dim)
        if values.shape != expected_shape:
            raise ValueError(
                f"expected layer vectors {expected_shape}, got {values.shape}"
            )

        self._data[self.completed] = values
        self._data.flush()
        with self.data_path.open("rb") as handle:
            os.fsync(handle.fileno())
        next_completed = self.completed + 1
        _atomic_json(
            self.progress_path,
            {"completed": next_completed},
        )
        self.completed = next_completed

    def layer_arrays(
        self,
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        if self.completed != self.total:
            raise ValueError(
                f"checkpoint is incomplete: {self.completed}/{self.total}"
            )
        positive = [
            self._data[: self.n_pos, layer, :]
            for layer in range(self.layers)
        ]
        negative = [
            self._data[self.n_pos :, layer, :]
            for layer in range(self.layers)
        ]
        return positive, negative

    def close(self) -> None:
        mmap = getattr(getattr(self, "_data", None), "_mmap", None)
        if mmap is not None:
            mmap.close()

    def remove(self) -> None:
        self.close()
        for path in (
            self.data_path,
            self.progress_path,
            self.meta_path,
            self.meta_path.with_suffix(".json.tmp"),
            self.progress_path.with_suffix(".json.tmp"),
        ):
            path.unlink(missing_ok=True)
        self.directory.rmdir()
