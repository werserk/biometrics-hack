import pickle

import torch
from torch import Tensor


class StyleGANModel:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Initialize the StyleGAN model for face generation.

        Args:
            model_path: Path to the pretrained StyleGAN model.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.G = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        with open(model_path, "rb") as f:
            G = pickle.load(f)["G_ema"].to(self.device)
        return G.eval()

    def generate_latents(self, batch_size: int = 8) -> Tensor:
        """Generate random latent vectors."""
        return torch.randn([batch_size, self.G.z_dim]).to(self.device)

    def synthesize_faces(self, latents: Tensor, truncation_psi: float = 0.5) -> Tensor:
        """
        Synthesize faces from latent vectors using the StyleGAN model.

        Args:
            latents: Latent vector(s) of shape (batch_size, G.z_dim).
            truncation_psi: Truncation trick value to control variation.

        Returns:
            Synthesized face images.
        """
        latents = latents.to(self.device)
        with torch.no_grad():
            c = None
            w = self.G.mapping(latents, c, truncation_psi=truncation_psi, truncation_cutoff=8)
            img = self.G.synthesis(w, noise_mode="const", force_fp32=True)
        return img

    @staticmethod
    def postprocess_faces(img: Tensor) -> Tensor:
        """
        Postprocess the StyleGAN images: convert from [-1, 1] to [0, 255] and change shape.

        Args:
            img: Tensor of images with shape (batch_size, 3, 1024, 1024).

        Returns:
            Post-processed images with shape (batch_size, 1024, 1024, 3).
        """
        img_processed = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img_processed
