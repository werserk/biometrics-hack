import numpy as np
import cv2

import torch

from app.face_reconstruction import FaceReconstructor
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_distances

class Hacker:
    def __init__(self, model_transposer, ):
        self.model_transposer = model_transposer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reconstructor = FaceReconstructor(root_dir=".", models_dir="./models", device=self.device)

        self.arcface = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def read_embedding(path_to_embedding: str) -> np.ndarray:
        embedding = np.load(path_to_embedding)

        return embedding

    def embedding_transposer(self):
        pass

    def generate_image(self, embedding: np.ndarray) -> np.ndarray:
        images = self.reconstructor.generate_images_by_embedding(
            torch.from_numpy(embedding).to(self.device), num_images=1
        )
        image = images[0]

        return image

    @staticmethod
    def save_image(path_to_save: str, image: np.ndarray):
        cv2.imwrite(path_to_save, image)

    def get_embedding(self, image: np.ndarray):
        faces = self.arcface.get(image)
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]

        return face['embedding']

    @staticmethod
    def embeddings_diff(true_embedding: np.ndarray, predicted_embedding: np.ndarray):
        cosine_dist = 1 - cosine_distances(true_embedding, predicted_embedding)
        return cosine_dist

    def metric(self, path_to_embedding: str):
        true_embedding = self.read_embedding(path_to_embedding)
        transposer_embedding = self.embedding_transposer(true_embedding)
        image = self.generate_image(transposer_embedding)
        predict_embedding = self.get_embedding(image)
        cosine_dist = self.embeddings_diff(true_embedding, predict_embedding)

        return cosine_dist



if __name__ == "__main__":
    model_transposer =
    hacker = Hacker()
