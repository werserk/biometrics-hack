import numpy as np
import cv2

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.face_reconstruction import FaceReconstructor
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_distances
from model_train.models import ComplexVectorModel, TransformerModel


class Hacker:
    def __init__(
        self,
        model_transposer,
    ):
        self.model_transposer = model_transposer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reconstructor = FaceReconstructor(root_dir=".", models_dir="./models", device=self.device)

        self.arcface = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def read_embedding(path_to_embedding: str) -> np.ndarray:
        embedding = np.load(path_to_embedding)

        return embedding

    def embedding_transposer(self, embedding: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.Tensor([embedding])
            print(tensor.shape)
            return self.model_transposer(tensor).numpy()

    def generate_image(self, embedding: np.ndarray) -> np.ndarray:

        images = self.reconstructor.generate_images_by_embedding(
            self.reconstructor.prepare_id_embedding(torch.from_numpy(embedding).to(self.device)), num_images=4
        )
        # image = images[0]

        return images

    @staticmethod
    def save_image(path_to_save: str, image: np.ndarray):
        cv2.imwrite(path_to_save, image)

    def get_embedding(self, image: np.ndarray):
        faces = self.arcface.get(image)
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]

        return face["embedding"]

    @staticmethod
    def embeddings_diff(true_embedding: np.ndarray, predicted_embedding: np.ndarray):
        print(true_embedding.shape, predicted_embedding.shape)
        cosine_dist = cosine_distances([true_embedding], [predicted_embedding])
        print("COS", cosine_dist)
        return cosine_dist

    def metric(self, path_to_embedding):
        # true_embedding = self.read_embedding(path_to_embedding)
        true_embedding = path_to_embedding
        transposer_embedding = self.embedding_transposer(true_embedding)
        images = self.generate_image(transposer_embedding)
        cos = []
        for image in images:
            try:
                self.save_image("image_res.jpg", image)
                predict_embedding = self.get_embedding(image)
                cosine_dist = self.embeddings_diff(true_embedding, predict_embedding)
                cos.append(cosine_dist[0][0])
            except IndexError:
                print("error")
                continue
        if len(cos) > 0:
            return np.max(cos)
        return "erorr"


if __name__ == "__main__":
    model_transposer = ComplexVectorModel(input_dim=512, output_dim=512)
    input_dim = 512  # Размерность входного вектора
    output_dim = 512  # Размерность выходного вектора
    nhead = 8  # Количество голов в self-attention
    num_encoder_layers = 6
    num_decoder_layers = 6
    learning_rate = 1e-4
    epochs = 50
    batch_size = 32

    model_transposer = ComplexVectorModel(input_dim=512, output_dim=512)
    model_transposer = TransformerModel(
        input_dim=input_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    model_filename = "/home/blogerlu/biometrics/biometrics-hack/model_epoch_10_train_0.0809_val_0.1280.pth"  # Укажи путь к файлу с весами
    model_filename = "/home/blogerlu/biometrics/biometrics-hack/model_epoch_30_train_0.0296_val_0.1138.pth"
    model_transposer.load_state_dict(torch.load(model_filename, map_location=torch.device("cpu")))

    # Перевод модели в режим оценки
    model_transposer.eval()

    hacker = Hacker(model_transposer)
    import pickle

    # Открываем файл .pickle
    with open("/mnt/sda1/hackathons/biometrics-hack/embedding_v3.pickle", "rb") as file:
        # Загружаем данные из файла
        vectors = pickle.load(file)

    # Предполагаем, что vectors — это список или массив векторов
    # Проходим по каждому вектору
    cs = []
    for vector, nm in vectors.items():
        # print(nm)
        try:
            cos = hacker.metric(nm["embedding"])
            if cos != "erorr":
                cs.append(cos)
        except IndexError:
            print("error")
    print(cs)
    print(np.mean(cs))
