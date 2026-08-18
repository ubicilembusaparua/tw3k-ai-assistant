import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"
ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def _is_file_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _required_files_ready(model_dir: Path) -> bool:
    return all(
        _is_file_ready(model_dir / filename)
        for filename in ("tokenizer.json", "model.onnx")
    )


def download(repo: str | None = None, dest: str | Path = "models") -> Path:
    """Download the selected ONNX model into ``models/<model-name>``.

    Complete caches are returned before contacting Hugging Face. Partial
    caches retain their existing files and only fetch what is missing.
    """

    repo = repo or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    model_dir = Path(dest) / repo
    tokenizer_dest = model_dir / "tokenizer.json"
    model_dest = model_dir / "model.onnx"

    if _required_files_ready(model_dir):
        print(f"Reusing complete model files at {model_dir}")
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = list_repo_files(repo_id=repo)
        onnx_file = next((candidate for candidate in ONNX_CANDIDATES if candidate in files), None)
        if not onnx_file:
            raise FileNotFoundError(f"No ONNX model found in Hugging Face repository {repo!r}")

        for remote, destination in (
            ("tokenizer.json", tokenizer_dest),
            (onnx_file, model_dest),
        ):
            if _is_file_ready(destination):
                print(f"  reused {destination}")
                continue
            source = hf_hub_download(repo_id=repo, filename=remote)
            shutil.copy2(source, destination)
            print(f"  saved {destination}")

        onnx_sidecar = onnx_file + "_data"
        sidecar_dest = model_dir / "model.onnx_data"
        if onnx_sidecar in files and not _is_file_ready(sidecar_dest):
            source = hf_hub_download(repo_id=repo, filename=onnx_sidecar)
            shutil.copy2(source, sidecar_dest)
            print(f"  saved {sidecar_dest}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare embedding model {repo!r} in {model_dir}: {exc}"
        ) from exc

    if not _required_files_ready(model_dir):
        raise RuntimeError(
            f"Embedding model {repo!r} is incomplete in {model_dir}; "
            "tokenizer.json and model.onnx are required."
        )

    print(f"Model ready at {model_dir}")
    return model_dir


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Hugging Face model ID (defaults to EMBEDDING_MODEL)")
    parser.add_argument("--dest", default="models", help="Root model directory")
    args = parser.parse_args()
    download(repo=args.model, dest=args.dest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
