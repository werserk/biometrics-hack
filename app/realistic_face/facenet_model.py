from typing import Union

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch import Tensor


class FaceNetModel:
    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize FaceNet for face detection, alignment, and embedding extraction.

        Args:
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=True)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def extract_embedding(self, img: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Extract FaceNet embedding for a given image.

        Args:
            img: Input image as a PyTorch tensor or NumPy array.

        Returns:
            FaceNet embeddings of the image.
        """
        if isinstance(img, np.ndarray):
            img = torch.Tensor(img)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            aligned = self.mtcnn(img.cpu())
            aligned_stack = torch.cat([face.unsqueeze(0) for face in aligned if face is not None], dim=0)
            embedding = self.resnet(aligned_stack.to(self.device))
        return embedding
