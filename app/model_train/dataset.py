import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):
    def __init__(self, input_dir, output_dir, split="train"):
        self.input_dir = os.path.join(input_dir, split)
        self.output_dir = os.path.join(output_dir, split)
        self.files = os.listdir(self.output_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.files[idx])
        output_path = os.path.join(self.output_dir, self.files[idx])

        # Загрузка векторов из файлов .npy
        input_vector = np.load(input_path)
        output_vector = np.load(output_path)

        # Преобразование в тензоры
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        output_tensor = torch.tensor(output_vector, dtype=torch.float32)

        return input_tensor, output_tensor
