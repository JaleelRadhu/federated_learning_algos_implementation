import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimizers.base_server import FedServerWithOptimizer
from optimizers.fedavg import FedAvgClient

class FedAMSGradClient(FedAvgClient):
    """
    Client for FedAMSGrad. It's identical to FedAvgClient as it performs local
    training with SGD and sends its updated model to the server.
    """
    pass

class FedAMSGradServer(FedServerWithOptimizer):
    """
    Server for FedAMSGrad.
    It maintains a server-side Adam optimizer with the AMSGrad variant enabled.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float, # Client LR
        device: torch.device,
        # Adam/AMSGrad specific parameters
        eta: float = 1e-3, # Server-side learning rate
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-8, # Epsilon term
    ):
        """
        Initializes the FedAMSGradServer.

        Args:
            model (nn.Module): The global PyTorch model.
            test_loader (DataLoader): DataLoader for the central test set.
            learning_rate (float): Client-side learning rate.
            device (torch.device): The device to run the model on.
            eta (float): Server-side learning rate for Adam.
            beta1 (float): Adam optimizer beta1.
            beta2 (float): Adam optimizer beta2.
            tau (float): Regularization/smoothing term for Adam (eps).
        """
        super().__init__(model, test_loader, learning_rate, device)
        
        # Server-side optimizer state with amsgrad=True
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=eta, betas=(beta1, beta2), eps=tau, amsgrad=True
        )