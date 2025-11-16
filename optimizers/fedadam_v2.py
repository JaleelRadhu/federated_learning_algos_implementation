import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimizers.base_server import FedServerWithOptimizer
from optimizers.fedavg import FedAvgClient

class FedAdamV2Client(FedAvgClient):
    """
    Client for FedAdam (Method 2). It's identical to FedAvgClient but we define it
    for clarity in the algorithm registration. It performs local training with SGD
    and sends its updated model to the server.
    """
    pass

class FedAdamV2Server(FedServerWithOptimizer):
    """
    Server for FedAdam (Method 2).
    It maintains a server-side Adam optimizer and applies aggregated client
    updates to the global model.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
        # FedAdam specific parameters
        beta1: float = 0.9,
        beta2: float = 0.999,
        eta: float = 1e-2, # Server-side learning rate
        tau: float = 1e-3, # Regularization term
    ):
        """
        Initializes the FedAdamV2Server.

        Args:
            model (nn.Module): The global PyTorch model.
            test_loader (DataLoader): DataLoader for the central test set.
            learning_rate (float): Client-side learning rate (passed but not used by server optimizer).
            device (torch.device): The device to run the model on.
            beta1 (float): Adam optimizer beta1.
            beta2 (float): Adam optimizer beta2.
            eta (float): Server-side learning rate for Adam.
            tau (float): Regularization/smoothing term for Adam.
        """
        super().__init__(model, test_loader, learning_rate, device)
        # Server-side optimizer state
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=eta,
            betas=(beta1, beta2),
            eps=tau
        )