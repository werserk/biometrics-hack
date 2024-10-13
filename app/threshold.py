import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from insightface.app import FaceAnalysis
import os
import cv2


class Threshold:
    def __init__(self, model, data):

        self.model = model
        self.data = data
        self.data_types = ["test", "train", "val"]

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        faces = self.model.get(image)
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        return face["embedding"]

    @staticmethod
    def read_image(path_to_image: str) -> np.ndarray:
        print(path_to_image)
        image = cv2.imread(path_to_image)
        return image

    def get_path_to_image(self, row: pd.DataFrame, path_to_folder: str) -> np.ndarray:
        tp = self.data_types[np.argmax([row[dt.replace("val", "valid")] for dt in self.data_types])]

        path_to_image = os.path.join(path_to_folder, tp, f"{row['attachment_id']}.jpg")

        return path_to_image

    def process_user_data(self, path_to_folder: str):

        grouped = self.data.groupby("user_id")
        results = []

        for user_id, group in grouped:

            if len(group) < 2:
                #                results.append({'user_id': user_id, 'average_cosine_distance': -100})
                continue
            try:
                vectors = np.array(
                    [
                        self.get_embedding(self.read_image(self.get_path_to_image(row, path_to_folder)))
                        for _, row in group.iterrows()
                    ]
                )
            except IndexError:
                print("error")
                continue
            #            print(vectors)
            if len(vectors) < 2:
                avg_distance = -100
            else:
                cosine_dist = cosine_distances(vectors)

                upper_triangle_indices = np.triu_indices(len(vectors), k=1)
                pairwise_distances = cosine_dist[upper_triangle_indices]

                avg_distance = 1 - np.mean(pairwise_distances)

            results.append({"user_id": user_id, "average_cosine_distance": avg_distance})
            dt = pd.DataFrame(results)
            dt.to_csv("/home/blogerlu/biometrics/biometrics-hack/meta_res.csv", index=False)

        return pd.DataFrame(results)


if __name__ == "__main__":

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    data = pd.read_csv("/mnt/sda1/hackathons/biometrics-hack/archive/annotations/meta/meta.csv")
    trch = Threshold(app, data)
    dt = trch.process_user_data("/mnt/sda1/hackathons/biometrics-hack/archive/images")
    dt.to_csv("../meta_res.csv", index=False)
