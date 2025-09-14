import torch
import torch.nn as nn

class AppleGrasper(nn.Module):
    def __init__(self):
        super(AppleGrasper, self).__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2)
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 8, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2232, out_features=64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return self.classifier(x)

