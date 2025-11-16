import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    adapted for FEMNIST.
    This architecture is based on  PyTorch MNIST example.
    """
    def __init__(self, num_classes=62):
        super(CNN, self).__init__()
        # FEMNIST images are 1x28x28
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # The input features for the first fully connected layer (fc1) is calculated as:
        # After conv1: 28-3+1 = 26x26. After conv2: 26-3+1 = 24x24.
        # After max_pool2d: 24/2 = 12x12.
        # Flattened size: 64 * 12 * 12 = 9216.
        self.fc1 = nn.Linear(9216, 128)
        # FEMNIST has 62 classes (10 digits + 26 lowercase + 26 uppercase).
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        # We return the raw logits. The loss function (e.g., CrossEntropyLoss) will apply softmax.
        return logits