import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DiscardCNN(nn.Module):
    def __init__(self, num_layers = 10, input_features=34, input_channels=29, output_features=34):
        super(MyCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=3, padding=1))
        
        for _ in range(1, num_layers - 1):
            self.conv_layers.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        
        self.conv_layers.append(nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, padding=0))
        
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        self.fc = nn.Linear(input_features, output_features)
    def forward(self, x):
        if x.dim() == 2:  # 如果輸入只有 (input_features, input_channels)
            x = x.unsqueeze(0)  # 變成 (1, input_features, input_channels)

        for i, conv in enumerate(self.conv_layers):
            if(i < len(self.conv_layers) - 1):
                x = self.activation(conv(x))
            else:
                x = conv(x)
                
        x = x.squeeze(1)
        x = self.fc(x)
        x = x.squeeze(0)
        #print("x.shape:",x.shape)
        
        return x

class MyCNN(nn.Module):
    def __init__(self, num_layers = 10, input_features=34, input_channels=29, output_features=1):
        super(MyCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=3, padding=1))
        
        for _ in range(1, num_layers - 1):
            self.conv_layers.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        
        self.conv_layers.append(nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, padding=0))
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.fc = nn.Linear(input_features, output_features)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        if x.dim() == 2:  # 如果輸入只有 (input_features, input_channels)
            x = x.unsqueeze(0)  # 變成 (1, input_features, input_channels)

        for i, conv in enumerate(self.conv_layers):
            if(i < len(self.conv_layers) - 1):
                x = self.activation(conv(x))
            else:
                x = conv(x)
                
        x = x.squeeze(1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x.squeeze(1)

class MyGRU(nn.Module):
    def __init__(self, input_size=288, hidden_size=64, num_layers=2, num_classes=1, sequence_length=1):
        super(MyGRU, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, hidden_size * sequence_length)
        self.fc2 = nn.Linear(hidden_size * sequence_length, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda")
        out,_ = self.gru(x, h0)
        out = out[:, -1, :]  # 取最後一個時間步的輸出
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
class MyDataset(Dataset):
    def __init__(self, file_path, chunk_size):
        self.file_path = file_path
        self.chunk_size = chunk_size
        
        self.columns = pd.read_csv(self.file_path, nrows=0).columns.str.strip().str.replace('\xa0', ' ').tolist()
        self.train_columns = [col for col in self.columns if col != 'discard']  # 排除 'discard' 欄位
        self.num_samples = sum(1 for _ in open(self.file_path, encoding='utf-8')) - 1  # 計算總樣本數
        
    def __getitem__(self, idx):
        chunk_start = idx // self.chunk_size * self.chunk_size
        df = pd.read_csv(self.file_path, skiprows=chunk_start + 1, nrows=self.chunk_size, header=None, encoding='utf-8')  # 跳過標題和之前的行

        df.columns = self.columns
        
        if 'discard' not in df.columns:
            raise KeyError(f"Chunk starting at row {chunk_start + 1} does not contain 'Discard' column.")
        
            
        sample_idx = idx % self.chunk_size
        
        train_data = df[self.train_columns].iloc[sample_idx].values.astype("float32")
        train_data = train_data.reshape(34, 29)
        
        value_data = df['discard'].iloc[sample_idx]  # 提取 'Discard' 欄位
        
        return torch.tensor(train_data, dtype=torch.float32), torch.tensor(value_data, dtype=torch.long)
        
    def __len__(self):
        return self.num_samples