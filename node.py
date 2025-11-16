import torch
import torch.nn as nn

class Node:
    """
    A parent class for Server and Client nodes
    """
    def __init__(self, model, device: torch.device, learning_rate: float):
        """
        Initializes the Node with a model, device, and learning rate.

        Args:
            model (nn.Module): The PyTorch model.
            device (torch.device): The device to run the model on.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.model = model
        self.device = device
        self.lr = learning_rate