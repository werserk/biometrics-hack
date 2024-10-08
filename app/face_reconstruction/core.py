import os
import warnings
from typing import List

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from Arc2Face.arc2face import CLIPTextModelWrapper, project_face_embs
from app.face_reconstruction.download import load_models

load_dotenv()
warnings.filterwarnings("ignore")


class FaceReconstructor:
    BASE_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    BETTER_SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, root_dir: str, models_dir: str, device: str = "cuda", download_models: bool = False) -> None:
        if download_models:
            load_models(dist_dir=models_dir)
        encoder = CLIPTextModelWrapper.from_pretrained(models_dir, subfolder="encoder", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(models_dir, subfolder="arc2face", torch_dtype=torch.float16)
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            FaceReconstructor.BASE_SD_MODEL,
            text_encoder=encoder,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
            # use_auth_token=os.getenv("HF_TOKEN"),
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to(device)
        self.model = FaceAnalysis(
            name="antelopev2", root=root_dir, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def image2embedding(self, image: np.array) -> torch.Tensor:
        faces = self.model.get(image)
        faces = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # select largest face (if more than one detected)
        id_embedding = torch.tensor(faces["embedding"], dtype=torch.float16)[None].cuda()
        id_embedding = id_embedding / torch.norm(id_embedding, dim=1, keepdim=True)  # normalize embedding
        id_embedding = project_face_embs(self.pipeline, id_embedding)  # pass through the encoder
        return id_embedding

    def generate_images_by_embedding(self, embedding: torch.Tensor, num_images: int) -> List[np.array]:
        images = self.pipeline(
            prompt_embeds=embedding, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images
        ).images
        return [np.array(image) for image in images]

    def generate_similar_images(self, image: np.array, num_images: int) -> List[np.array]:
        return self.generate_images_by_embedding(self.image2embedding(image), num_images)
