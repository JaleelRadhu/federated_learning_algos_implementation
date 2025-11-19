import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List

from node import Node

class FedServerWithOptimizer(Node):
    """
    A base class for federated servers that use a server-side optimizer.
    This class handles the common logic for model aggregation, pseudo-gradient
    calculation, model update, and testing.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
    ):
        model.to(device)
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.test_loader = test_loader
        self.optimizer = None

    def update_model(self, client_state_dicts: List[OrderedDict], client_weights: List[float]):
        """
        Updates the global model using the server-side optimizer.
        """
        if not client_state_dicts or self.optimizer is None:
            return

        initial_global_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Perform weighted averaging of client models
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        aggregated_state_dict = OrderedDict()
        for key in initial_global_state.keys():
            weighted_sum = torch.stack([state_dict[key].to(self.device) * weight for state_dict, weight in zip(client_state_dicts, normalized_weights)]).sum(dim=0)
            aggregated_state_dict[key] = weighted_sum

        model_delta = OrderedDict()
        for key in initial_global_state.keys():
            model_delta[key] = aggregated_state_dict[key] - initial_global_state[key]

        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = -1.0 * model_delta[name]

        # Take a step with the server-side optimizer
        self.optimizer.step()

    def test(self):
        """
        Evaluates the global model on the central test set.
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch['image'].to(self.device), batch['character'].to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return avg_loss, accuracy