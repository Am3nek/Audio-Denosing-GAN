import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            # First convolution layer (input -> 64 features)
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Second convolution layer (64 -> 128 features)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Third convolution layer (128 -> 256 features)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Fourth convolution layer (256 -> 512 features)
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Global Average Pooling to reduce dimensions
            nn.AdaptiveAvgPool1d(1),

            # Flatten and Fully Connected Layer
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output probability (0: Fake, 1: Real)
        )

    def forward(self, x):
        return self.model(x)
