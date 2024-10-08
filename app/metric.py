from typing import Union

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))


ImageType = Union[np.ndarray, str]


class Similarity:
    def __init__(self) -> None:
        self.model = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image: np.array) -> np.ndarray:
        faces = self.model.get(image)
        return faces[0].normed_embedding

    def _to_embedding(self, obj: ImageType) -> np.ndarray:
        if isinstance(obj, str):
            obj = cv2.imread(obj)
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 1 and obj.shape[0] == 512:
                return obj
            obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
            return self.get_embedding(obj)
        raise TypeError(f"Expected np.ndarray or str, got {type(obj)}")

    def compare(self, obj1: ImageType, obj2: ImageType) -> float:
        embedding1 = self._to_embedding(obj1)
        embedding2 = self._to_embedding(obj2)
        return cosine_similarity(embedding1, embedding2)

    def __call__(self, obj1: ImageType, obj2: ImageType) -> float:
        return self.compare(obj1, obj2)
