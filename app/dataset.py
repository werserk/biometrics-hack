import cv2
import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, model):
        self.model = model
        # self.df = pd.DataFrame(['attachment_id', 'user_id', 'type'])
        self.data_types = ['test', 'train', 'val']

    @staticmethod
    def save_embedding(path_to_save: str, embedding: np.ndarray):
        np.save(path_to_save, embedding)

    @staticmethod
    def read_image(path_to_image: str) -> np.ndarray:
        image = cv2.imread(path_to_image)

        return image

    def create_folders(self, path_to_folder: str):
        os.mkdir(path_to_folder)
        for type in self.data_types:
            os.mkdir(os.path.join(path_to_folder, type))

    def generate_dataset(self, data: pd.DataFrame, root_path: str, path_to_save: str):

        for _, row in data.iterrows():
            attachment_id = row['attachment_id']
            user_id = row['user_id']
            type = self.data_types[np.argmax([row[dt] for dt in self.data_types])]

            path_to_save_embedding = os.path.join(path_to_save, type, f'{attachment_id}.npy')

            path_to_image = os.path.join(root_path, type, attachment_id + '.jpg')
            image = self.read_image(path_to_image)
            embedding = self.model.predict(image)
            self.save_embedding(path_to_save_embedding, embedding)


    def get_embedding(self, image: np.array) -> np.ndarray:
        faces = self.model.get(image)
        return faces[0].normed_embedding

