import torch.nn as nn
import torch

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