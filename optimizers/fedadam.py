import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from node import Node

class FedAdamClient(Node):
    """
    A client node that performs training based on the FedAvg algorithm, but uses the
    Adam optimizer for local training instead of SGD.
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
        Initializes the FedAdamClient.

        Args:
            client_id (int): A unique identifier for the client.
            model (nn.Module): The PyTorch model.
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
        using the Adam optimizer.

        Args:
            local_epochs (int): The number of epochs to train locally.

        Returns:
            float: The average training loss over all local epochs.
        """
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        for epoch in range(local_epochs):
            # Add a progress bar to visualize the training loop for each client
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