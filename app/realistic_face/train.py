# train.py

import torch

from app.realistic_face.face_reconstruction import FaceReconstruction
from app.realistic_face.facenet_model import FaceNetModel
from app.realistic_face.stylegan_model import StyleGANModel


def main():
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    stylegan = StyleGANModel(model_path="stylegan2-ada-pytorch/ffhq.pkl", device=device)
    facenet = FaceNetModel(device=device)

    # Initialize FaceReconstruction
    face_reconstructor = FaceReconstruction(stylegan, facenet)

    # Generate latent vectors and save them
    print("Starting latent vector generation...")
    face_reconstructor.generate_and_save_latents(n=160, batch_size=16, file_name="pregenerated.pt")
    print("Latent vectors and embeddings saved successfully.")


if __name__ == "__main__":
    main()
