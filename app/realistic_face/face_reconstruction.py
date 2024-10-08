from math import exp
from random import random
from typing import List, Tuple, Optional

import torch
from torch import Tensor

from app.realistic_face.facenet_model import FaceNetModel
from app.realistic_face.stylegan_model import StyleGANModel


class FaceReconstruction:
    def __init__(self, stylegan: StyleGANModel, facenet: FaceNetModel) -> None:
        """
        Initialize the FaceReconstruction class with pre-trained models.

        Args:
            stylegan: Pre-initialized StyleGAN model.
            facenet: Pre-initialized FaceNet model.
        """
        self.stylegan = stylegan
        self.facenet = facenet
        self.device = stylegan.device
        self.pregen_latents: Optional[Tensor] = None
        self.pregen_embeddings: Optional[Tensor] = None

    def generate_and_save_latents(
        self, n: int = 8000, batch_size: int = 16, file_name: str = "pregenerated.pt"
    ) -> None:
        """Generate and save latent vectors and their embeddings."""
        iterations = n // batch_size
        latents_list, embedding_list = [], []

        with torch.no_grad():
            for _ in range(iterations):
                latents = self.stylegan.generate_latents(batch_size)
                faces = self.stylegan.synthesize_faces(latents)
                processed_faces = self.stylegan.postprocess_faces(faces)
                embeddings = self.facenet.extract_embedding(processed_faces)

                latents_list.append(latents.cpu())
                embedding_list.append(embeddings.cpu())

            all_latents = torch.cat(latents_list)
            all_embeddings = torch.cat(embedding_list)
        torch.save([all_latents, all_embeddings], file_name)

    def load_pregenerated_latents(self, path_prefix: str = "pregenerated_") -> None:
        """Load pregenerated latent vectors and embeddings."""
        latents, embeddings = [], []
        for i in range(20):
            latent, embedding = torch.load(f"{path_prefix}{i}.pt")
            latents.append(latent)
            embeddings.append(embedding)
        self.pregen_latents = torch.cat(latents)
        self.pregen_embeddings = torch.cat(embeddings)

    def find_closest_latent(self, target_emb: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """
        Find the closest pregenerated latent vector to the target embedding.

        Args:
            target_emb: Target FaceNet embedding to find the closest latent for.
            offset: Index offset to find the nth closest match.

        Returns:
            A tuple of (latent vector, corresponding embedding).
        """
        if self.pregen_latents is None or self.pregen_embeddings is None:
            raise ValueError("Pregenerated latents and embeddings are not loaded")

        norms = (self.pregen_embeddings - target_emb.to(self.pregen_embeddings)).norm(dim=1)
        best_idx = norms.argsort()[offset]
        return self.pregen_latents[best_idx], self.pregen_embeddings[best_idx]

    def perform_face_reconstruction(
        self,
        target_emb: Tensor,
        pregen: bool = True,
        pregen_offset: int = 0,
        init_zeros: bool = False,
        iters: int = 400,
        use_annealing: bool = False,
        std_multiplier: float = 0.98,
    ) -> Tuple[Tensor, List[Tensor], List[float], List[Tensor], List[Tensor]]:
        """
        Perform face reconstruction to match a target embedding as closely as possible.

        Args:
            target_emb: Target embedding (shape: (1, 512)).
            pregen: Whether to start from the closest pregenerated latent (default: True).
            pregen_offset: Offset for using n-th closest pregenerated match.
            init_zeros: If pregen is False, start from a zero latent vector or random noise.
            iters: Number of iterations for optimization.
            use_annealing: Whether to use simulated annealing.
            std_multiplier: Multiplier for noise standard deviation during optimization.

        Returns:
            Tuple of the best latent vector, the list of best images, the list of best norms,
            the list of best latent vectors, and the list of best embeddings at each improvement.
        """

        def safe_exp(x: float) -> float:
            """Safely compute exponential."""
            try:
                return exp(x)
            except OverflowError:
                return 0

        def P(e: float, e_prime: float, T: float) -> float:
            """Simulated annealing probability function."""
            if e_prime < e:
                return 1
            else:
                return safe_exp(-(e_prime - e) / T)

        target_emb = target_emb.to(self.device)

        # Initialize latent vector
        if init_zeros:
            best_latent = torch.zeros([1, self.stylegan.G.z_dim]).to(self.device)
        else:
            best_latent = torch.randn([1, self.stylegan.G.z_dim]).to(self.device)

        best_norm = float("inf")

        # Use pregenerated latent if available
        if pregen:
            if self.pregen_latents is None or self.pregen_embeddings is None:
                raise ValueError("Pregenerated latents and embeddings are not loaded")
            best_latent, best_embedding = self.find_closest_latent(target_emb, pregen_offset)
            best_norm = (target_emb - best_embedding).norm().item()

        current_latent = best_latent
        current_norm = best_norm

        best_list, best_norm_list, best_latent_list, best_emb_list = [], [], [], []

        # Add the image of the starting latent
        with torch.no_grad():
            best_face = self.stylegan.postprocess_faces(self.stylegan.synthesize_faces(current_latent))[0].cpu()
        best_list.append(best_face)
        best_norm_list.append(best_norm)
        best_latent_list.append(best_latent)
        best_emb_list.append(self.facenet.extract_embedding(best_face))

        print(f"Starting norm: {best_norm}")
        std = 1.0
        T = 0.0

        # Iterative optimization loop
        with torch.no_grad():
            for i in range(iters):
                if use_annealing:
                    T = 1 - (i + 1) / iters

                neighbor_latents = current_latent + std * torch.randn([16, self.stylegan.G.z_dim]).to(self.device)
                faces = self.stylegan.synthesize_faces(neighbor_latents)
                faces_pp = self.stylegan.postprocess_faces(faces)
                embeddings = self.facenet.extract_embedding(faces_pp)

                norms = (embeddings - target_emb).norm(dim=1)
                best_idx = torch.argmin(norms)

                if P(current_norm, norms[best_idx].item(), T) > random():
                    current_latent = neighbor_latents[best_idx]
                    current_norm = norms[best_idx].item()

                if norms[best_idx] < best_norm:
                    best_latent = neighbor_latents[best_idx]
                    best_norm = norms[best_idx].item()
                    best_list.append(faces_pp[best_idx].cpu())
                    best_norm_list.append(best_norm)
                    best_latent_list.append(best_latent)
                    best_emb_list.append(embeddings[best_idx])

                std *= std_multiplier

        return best_latent, best_list, best_norm_list, best_latent_list, best_emb_list
