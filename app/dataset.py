import cv2
import pandas as pd
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy.spatial.distance import cosine, euclidean, cityblock
#%%

class Dataset:
    def __init__(self, model):
        self.model = model
        # self.df = pd.DataFrame(['attachment_id', 'user_id', 'type'])
        self.data_types = ['test', 'train', 'valid']

    @staticmethod
    def save_embedding(path_to_save: str, embedding: np.ndarray):
        np.save(path_to_save, embedding)

    @staticmethod
    def read_image(path_to_image: str) -> np.ndarray:
        image = cv2.imread(path_to_image)

        return image

    def create_folders(self, path_to_folder: str):
        os.mkdir(path_to_folder)
        for tp in self.data_types:
            os.mkdir(os.path.join(path_to_folder, tp.replace('valid', 'val')))

    def generate_dataset(self, data: pd.DataFrame, root_path: str, path_to_save: str):
        self.create_folders(path_to_save)
        for _, row in data.iterrows():
            attachment_id = row['attachment_id']
            user_id = row['user_id']
            tp = self.data_types[np.argmax([row[dt] for dt in self.data_types])].replace('valid', 'val')

            path_to_save_embedding = os.path.join(path_to_save, tp, f'{attachment_id}.npy')

            path_to_image = os.path.join(root_path, tp, attachment_id + '.jpg')
            image = self.read_image(path_to_image)
 #           print(type(image))
#            print(image)
            if image is None:
               print(f'image not find: {attachment_id}, {user_id}, {tp}')
               continue
            try:
               embedding = self.get_embedding(image)
            except IndexError:
               print(f'face not found: {attachment_id}, {user_id}, {tp}')            
            self.save_embedding(path_to_save_embedding, embedding)


    def get_embedding(self, image: np.array) -> np.ndarray:
        faces = self.model.get(image)
        return faces[0].normed_embedding



if __name__ == '__main__':
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    data = pd.read_csv('/mnt/sda1/hackathons/biometrics-hack/archive/annotations/meta/meta.csv')

    dataset = Dataset(app)
    dataset.generate_dataset(data, '/mnt/sda1/hackathons/biometrics-hack/archive/images', '/home/blogerlu/biometrics/biometrics-hack/embeddings')
