import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimizers.base_server import FedServerWithOptimizer
from optimizers.fedavg import FedAvgClient

class FedAdagradClient(FedAvgClient):
    """
    Client for FedAdagrad. It's identical to FedAvgClient as it performs local
    training with SGD and sends its updated model to the server.
    """
    pass

class FedAdagradServer(FedServerWithOptimizer):
    """
    Server for FedAdagrad.
    It maintains a server-side Adagrad optimizer and applies aggregated client
    updates to the global model.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float, 
        device: torch.device,
        # FedAdagrad specific parameters
        eta: float = 1e-2, # Server-side learning rate
        tau: float = 1e-9, # avoid division by zero
    ):
        """
        Initializes the FedAdagradServer.

        Args:
            model (nn.Module): The global PyTorch model.
            test_loader (DataLoader): DataLoader for the central test set.
            learning_rate (float): Client-side learning rate (not used by server optimizer).
            device (torch.device): The device to run the model on.
            eta (float): Server-side learning rate for Adagrad.
            tau (float): Regularization/smoothing term for Adagrad (eps).
        """
        super().__init__(model, test_loader, learning_rate, device)
        # Server-side optimizer state
        self.optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=eta,
            eps=tau
        )