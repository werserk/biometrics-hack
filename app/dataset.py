import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from insightface.app import FaceAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.face_reconstruction import FaceReconstructor


class Dateset:
    def __init__(self):
        self.data_types = ["test", "train", "val"]

    @staticmethod
    def create_folder(path_to_folder):
        try:
            os.mkdir(path_to_folder)
        except FileExistsError:
            print(f"file {path_to_folder} already exists ")

    def create_folders(self, path_to_folder: str):

        self.create_folder(path_to_folder)
        for tp in self.data_types:
            self.create_folder(os.path.join(path_to_folder, tp))

    @staticmethod
    def file_check(path_to_file: str):
        dirname = os.path.dirname(path_to_file)
        filename = os.path.basename(path_to_file)
        if filename in os.listdir(dirname):
            return True
        return False


class DatasetI2E(Dateset):
    def __init__(self, model):
        super().__init__()
        self.data_types = ["test", "train", "val"]
        self.model = model  # %%

    @staticmethod
    def save_embedding(path_to_save: str, embedding: np.ndarray):
        np.save(path_to_save, embedding)

    @staticmethod
    def read_image(path_to_image: str) -> np.ndarray:
        image = cv2.imread(path_to_image)
        return image

    def get_embedding(self, image: np.array) -> np.ndarray:
        # faces = self.model.get(image)
        # face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        embedding = self.model.image2embedding(image)

        # return face['embedding']
        return embedding

    def generate_dataset(self, data: pd.DataFrame, root_path: str, path_to_save: str):
        """

        :param data:
        :param root_path: path to folder with train, test, valid folders
        :param path_to_save:
        :return:
        """
        self.create_folders(path_to_save)
        for _, row in data.iterrows():
            attachment_id = row["attachment_id"]
            user_id = row["user_id"]
            tp = self.data_types[np.argmax([row[dt.replace("val", "valid")] for dt in self.data_types])]

            path_to_save_embedding = os.path.join(path_to_save, tp, f"{attachment_id}.npy")
            if self.file_check(path_to_save_embedding):
                continue
            path_to_image = os.path.join(root_path, tp, attachment_id + ".jpg")

            image = self.read_image(path_to_image)

            if image is None:
                print(f"image not find: {attachment_id}, {user_id}, {tp}")
                continue

            try:
                embedding = self.get_embedding(image)
            except IndexError:
                print(f"face not found: {attachment_id}, {user_id}, {tp}")

            self.save_embedding(path_to_save_embedding, embedding)


class DatasetE2I(Dateset):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reconstructor = FaceReconstructor(root_dir=".", models_dir="./models", device=self.device)

    def get_image(self, embedding) -> np.ndarray:
        images = self.reconstructor.generate_images_by_embedding(
            torch.from_numpy(embedding).to(self.device), num_images=1
        )
        image = images[0]

        return image

    @staticmethod
    def read_embedding(path_to_embedding: str) -> np.ndarray:
        embedding = np.load(path_to_embedding)

        return embedding

    @staticmethod
    def save_image(path_to_save: str, image: np.ndarray):
        cv2.imwrite(path_to_save, image)

    def generate_dataset(self, data: pd.DataFrame, root_path: str, path_to_save: str):
        """

        :param data:
        :param root_path: path to folder with train, test, valid folders
        :param path_to_save:
        :return:
        """
        self.create_folders(path_to_save)
        for _, row in data.iterrows():
            attachment_id = row["attachment_id"]
            user_id = row["user_id"]
            tp = self.data_types[np.argmax([row[dt.replace("val", "valid")] for dt in self.data_types])]

            path_to_save_image = os.path.join(path_to_save, tp, f"{attachment_id}.npy")
            if self.file_check(path_to_save_image):
                continue

            path_to_embedding = os.path.join(root_path, tp, attachment_id + ".npy")

            embedding = self.read_embedding(path_to_embedding)

            if embedding is None:
                print(f"embedding not find: {attachment_id}, {user_id}, {tp}")
                continue
            try:
                image = self.get_image(embedding)
            except IndexError:
                print(f"vector error: {attachment_id}, {user_id}, {tp}")

            self.save_image(path_to_save_image, image)


if __name__ == "__main__":
    # app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # data = pd.read_csv('/mnt/sda1/hackathons/biometrics-hack/archive/annotations/meta/meta.csv')

    # dataset = Dataset(app)
    # dataset.generate_dataset(data, '/mnt/sda1/hackathons/biometrics-hack/archive/images', '/home/blogerlu/biometrics/biometrics-hack/embeddings')

    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument("mod", help="I2E/E2I")
    parser.add_argument("path_to_csv", help="path to csv file")
    parser.add_argument("root_path", help="path to images/embeddings [train, val, test]")
    parser.add_argument("path_to_save", help="path to save results")

    args = parser.parse_args()
    data = pd.read_csv(args.path_to_csv)

    if args.mod == "I2E":
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        reconstructor = FaceReconstructor(root_dir=".", models_dir="./models", device="cuda")
        with torch.no_grad():
            dataset = DatasetI2E(reconstructor)
            dataset.generate_dataset(data, args.root_path, args.path_to_save)

    elif args.mod == "E2I":
        dataset = DatasetE2I()
        dataset.generate_dataset(data, args.root_path, args.path_to_save)
