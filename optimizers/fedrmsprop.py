import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimizers.base_server import FedServerWithOptimizer
from optimizers.fedavg import FedAvgClient

class FedRMSPropClient(FedAvgClient):
    """
    Client for FedRMSProp. It's identical to FedAvgClient as it performs local
    training with SGD and sends its updated model to the server.
    """
    pass

class FedRMSPropServer(FedServerWithOptimizer):
    """
    Server for FedRMSProp.
    It maintains a server-side RMSprop optimizer and applies aggregated client
    updates to the global model.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float, # This is the client LR, passed for consistency
        device: torch.device,
        # FedRMSProp specific parameters
        eta: float = 1e-2, # Server-side learning rate
        alpha: float = 0.99, # Smoothing constant
        tau: float = 1e-2, # Regularization term
    ):
        """
        Initializes the FedRMSPropServer.

        Args:
            model (nn.Module): The global PyTorch model.
            test_loader (DataLoader): DataLoader for the central test set.
            learning_rate (float): Client-side learning rate (not used by server optimizer).
            device (torch.device): The device to run the model on.
            eta (float): Server-side learning rate for RMSprop.
            alpha (float): RMSprop smoothing constant.
            tau (float): Regularization/smoothing term for RMSprop (eps).
        """
        super().__init__(model, test_loader, learning_rate, device)
        # Server-side optimizer state
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=eta,
            alpha=alpha,
            eps=tau
        )