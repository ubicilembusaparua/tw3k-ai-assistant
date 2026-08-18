import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]
DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"


class Embedder:
    """Fast local ONNX embedding engine using tokenizers and ONNX Runtime."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        models_dir: Union[str, Path] = "models",
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.dest_dir = Path(models_dir) / self.model_name
        self._ensure_model_downloaded()

        tokenizer_path = self.dest_dir / "tokenizer.json"
        model_path = self.dest_dir / "model.onnx"

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def _ensure_model_downloaded(self):
        """Downloads tokenizer.json and model.onnx if not present locally."""
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dest = self.dest_dir / "tokenizer.json"
        model_dest = self.dest_dir / "model.onnx"

        if (
            tokenizer_dest.is_file()
            and tokenizer_dest.stat().st_size > 0
            and model_dest.is_file()
            and model_dest.stat().st_size > 0
        ):
            logging.info("Reusing complete model files at %s", self.dest_dir)
            return

        try:
            files = list_repo_files(repo_id=self.model_name)
            onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
            if not onnx_file:
                raise FileNotFoundError(
                    f"No ONNX model candidate found in HF repo {self.model_name}"
                )

            for remote, local in (
                ("tokenizer.json", tokenizer_dest),
                (onnx_file, model_dest),
            ):
                if local.is_file() and local.stat().st_size > 0:
                    continue
                source = hf_hub_download(repo_id=self.model_name, filename=remote)
                shutil.copy2(source, local)

            onnx_sidecar = onnx_file + "_data"
            sidecar_dest = self.dest_dir / "model.onnx_data"
            if onnx_sidecar in files and not sidecar_dest.is_file():
                source = hf_hub_download(repo_id=self.model_name, filename=onnx_sidecar)
                shutil.copy2(source, sidecar_dest)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare embedding model {self.model_name!r} "
                f"in {self.dest_dir}: {exc}"
            ) from exc

    def get_embedding_dimension(self) -> int:
        """Returns embedding vector dimension (384 for MiniLM-L6-v2)."""
        return 384

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string into a 1D embedding vector."""
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of text strings into a 2D array of embedding vectors."""
        if not texts:
            return np.empty((0, self.get_embedding_dimension()), dtype=np.float32)

        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)
        encoded = self.tokenizer.encode_batch(texts)

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)

        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norms, 1e-9)

        return pooled.astype(np.float32)
