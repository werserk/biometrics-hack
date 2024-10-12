
import torch.nn as nn

class VectorModel(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(VectorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ComplexVectorModel(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(ComplexVectorModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(input_dim, output_dim)
        
        self.output_fc = nn.Linear(output_dim, output_dim)
        
        self.transformer = nn.Transformer(d_model=output_dim, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)  # Добавляем ось для трансформера
        
        out = self.transformer(x, x)
        
        out = self.fc_out(out.squeeze(1))  # Убираем временную ось
        
        return out

