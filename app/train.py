import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model_train.dataset import VectorDataset
from model_train.models import VectorModel, ComplexVectorModel, TransformerModel

input_dir = "/home/blogerlu/biometrics/biometrics-hack/embeddings"  # Путь к папке с input
output_dir = "/home/blogerlu/biometrics/biometrics-hack/embeddings_antilopa"  # Путь к папке с output
input_dim = 512  # Размерность входных векторов, например
batch_size = 10
epochs = 200
learning_rate = 1e-3

# Датасеты и загрузчики данных
train_dataset = VectorDataset(input_dir, output_dir, split="train")
val_dataset = VectorDataset(input_dir, output_dir, split="val")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ComplexVectorModel(input_dim=input_dim)

input_dim = 512  # Размерность входного вектора
output_dim = 512  # Размерность выходного вектора
nhead = 8  # Количество голов в self-attention
num_encoder_layers = 4
num_decoder_layers = 4
learning_rate = 1e-4
epochs = 50
batch_size = 128


model = TransformerModel(
    input_dim=input_dim,
    output_dim=output_dim,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
)

criterion = nn.MSELoss()
criterion = nn.CosineEmbeddingLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# Обучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()
        #        print(inputs.shape)
        # Прямой проход
        outputs = model(inputs)
        #       print(outputs.shape)
        #    print(targets.shape)
        # Вычисление потерь
        loss = criterion(outputs, targets, torch.ones(targets.shape[0]).cuda())

        # Обратный проход и обновление весов
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Вывод статистики
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Валидация модели
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, torch.ones(targets.shape[0]).cuda())
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")

    if (epoch + 1) % 10 == 0:
        model_filename = f"model_epoch_{epoch+1}_train_{running_loss / len(train_loader):.4f}_val_{val_loss / len(val_loader):.4f}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    scheduler.step()

    # Вывод текущего learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(current_lr)
print("Обучение завершено!")
