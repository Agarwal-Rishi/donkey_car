import torch.nn as nn


steering_bins = 15
throttle_bins = 20
image_height = 120
image_width = 160
throttle_max = 0.8


class AutonomousDriver(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 10, 100),
            nn.ReLU(inplace=True),
        )

        self.steering_head = nn.Linear(100, steering_bins)
        self.throttle_head = nn.Linear(100, throttle_bins)

    def forward(self, x):
        x = self.features(x)
        steering_logits = self.steering_head(x)
        throttle_logits = self.throttle_head(x)
        return steering_logits, throttle_logits
