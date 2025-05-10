from torch import nn


class SimpleModel(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_shape: int):
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = None
        self.output_shape = output_shape
        
    def forward(self, x):
        x = self.convolutional_block_1(x)
        x = self.convolutional_block_2(x)
        x = self.flatten(x)
        if self.classifier is None:
            self.classifier = nn.Linear(
                in_features=x.shape[1],
                out_features=self.output_shape
            )
        x = self.classifier(x)
        return x
    
class BatchModel(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_shape: int):
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = None
        self.output_shape = output_shape
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            
    def forward(self, x):
        x = self.convolutional_block_1(x)
        x = self.convolutional_block_2(x)
        x = self.flatten(x)
        if self.classifier is None:
            self.classifier = nn.Linear(
                in_features=x.shape[1],
                out_features=self.output_shape
            )
        x = self.classifier(x)
        return x
        
        
# спробуємо додати дропаут та підвищити learning rate
class DropoutModel(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_shape: int):
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        self.flatten = nn.Flatten()
        self.classifier = None
        self.output_shape = output_shape
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            
    def forward(self, x):
        x = self.convolutional_block_1(x)
        x = self.convolutional_block_2(x)
        x = self.flatten(x)
        if self.classifier is None:
            self.classifier = nn.Linear(
                in_features=x.shape[1],
                out_features=self.output_shape
            )
        x = self.classifier(x)
        return x
    
    
class LeNetModel(nn.Module):
    def __init__(self, input_channels: int, output_shape: int):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
        )
        self.fully_connected_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=output_shape),
        )
        
    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.fully_connected_block(x)
        return x
        
class ConvNetModel(nn.Module):
    def __init__(self, input_channels: int, output_shape: int):
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.convolutional_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.fully_connected_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 3 * 3, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=output_shape),
        )
        
    def forward(self, x):
        x = self.convolutional_block_1(x)
        x = self.convolutional_block_2(x)
        x = self.convolutional_block_3(x)
        x = self.fully_connected_block(x)
        return x