import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
from typing import List

from node import Node

class FedAvgClient(Node):
    """
    A client node that performs training based on the Federated Averaging (FedAvg) algorithm.
    It uses Stochastic Gradient Descent (SGD) as its local optimizer.
    """
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float,
        device: torch.device,
    ):
        """
        Initializes the FedAvgClient.

        Args:
            client_id (int): A unique identifier for the client.
            model (CNN): The PyTorch model.
            dataloader (DataLoader): The client's local data loader.
            learning_rate (float): The learning rate for the optimizer.
            device (torch.device): The device to run the model on.
        """
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.client_id = client_id
        self.dataloader = dataloader

    def train(self, local_epochs: int):
        """
        Trains the client's local model for a specified number of local epochs
        using the SGD optimizer.

        Args:
            local_epochs (int): The number of epochs to train locally.

        Returns:
            float: The average training loss over all local epochs.
        """
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        for epoch in range(local_epochs):
            progress_bar = tqdm(self.dataloader, 
                                desc=f"Client {self.client_id} Epoch {epoch+1}/{local_epochs}", 
                                leave=False)
            for batch in progress_bar:
                images, labels = batch['image'].to(self.device), batch['character'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return total_loss / (len(self.dataloader) * local_epochs)

class FedAvgServer(Node):
    """
    A server that orchestrates the training process using the FedAvg algorithm.
    It performs weighted averaging of client models.
    """
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
    ):
        """
        Initializes the FedAvgServer.

        Args:
            model (nn.Module): The global PyTorch model.
            test_loader (DataLoader): The DataLoader for the central test set.
            learning_rate (float): The learning rate (can be used for server-side optimization).
            device (torch.device): The device to run the model on.
        """
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.test_loader = test_loader

    def update_model(self, client_state_dicts: List[OrderedDict], client_weights: List[float]):
        """
        Updates the global model by performing a weighted average of the client models.

        Args:
            client_state_dicts (List[OrderedDict]): A list of model state dictionaries from the clients.
            client_weights (List[float]): A list of weights corresponding to each client,
                                          typically proportional to the size of their local dataset.
        """
        if not client_state_dicts:
            return

        # Ensure client_weights sum to 1 for correct weighted average
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialize a new state_dict for the global model by copying the first client's state_dict
        # and multiplying by its weight. This avoids initializing with zeros.
        global_state_dict = OrderedDict()
        first_client_state_dict = client_state_dicts[0]
        first_weight = normalized_weights[0]
        for key in first_client_state_dict.keys():
            global_state_dict[key] = first_client_state_dict[key].to(self.device) * first_weight

        # Accumulate the weighted parameters from the rest of the clients
        for weight, state_dict in zip(normalized_weights[1:], client_state_dicts[1:]):
            for key in global_state_dict.keys():
                global_state_dict[key] += state_dict[key].to(self.device) * weight
        # Load the new state into the global model
        self.model.load_state_dict(global_state_dict)

    def test(self):
        """
        Evaluates the global model on the central test set.

        Returns:
            tuple: A tuple containing the average test loss and accuracy.
        """
        self.model.to(self.device)
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