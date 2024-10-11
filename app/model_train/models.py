
import torch.nn as nn

class VectorModel(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(VectorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # Скрытый слой
        self.fc2 = nn.Linear(1024, output_dim)  # Выходной слой
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x