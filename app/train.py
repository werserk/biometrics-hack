import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model_train.dataset import VectorDataset
from model_train.models import VectorModel

input_dir = '/dt/embeddings'  # Путь к папке с input
output_dir = '/dt/embeddings_antilop'  # Путь к папке с output
input_dim = 256  # Размерность входных векторов, например
batch_size = 2
epochs = 20
learning_rate = 1e-3

# Датасеты и загрузчики данных
train_dataset = VectorDataset(input_dir, output_dir, split='train')
val_dataset = VectorDataset(input_dir, output_dir, split='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = VectorModel(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)

        # Вычисление потерь
        loss = criterion(outputs, targets)

        # Обратный проход и обновление весов
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Вывод статистики
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

    # Валидация модели
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss / len(val_loader)}')

print("Обучение завершено!")