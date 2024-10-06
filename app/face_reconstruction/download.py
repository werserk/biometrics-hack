import os
import zipfile

import gdown
from huggingface_hub import hf_hub_download


def _load_arc2face(dist_dir: str) -> None:
    hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir=dist_dir)
    hf_hub_download(
        repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir=dist_dir
    )
    hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir=dist_dir)
    hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir=dist_dir)
    hf_hub_download(
        repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir=os.path.join(dist_dir, "antelopev2")
    )


def _load_antelopev2(dist_dir: str) -> None:
    file_id = "18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"
    destination = "antelopev2.zip"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(download_url, destination, quiet=False)

    with zipfile.ZipFile(destination, "r") as zip_ref:
        zip_ref.extractall(dist_dir)

    os.remove(destination)
    os.remove(os.path.join(dist_dir, "antelopev2", "glintr100.onnx"))


def load_models(dist_dir: str) -> None:
    _load_arc2face(dist_dir)
    _load_antelopev2(dist_dir)


__all__ = ["load_models"]
