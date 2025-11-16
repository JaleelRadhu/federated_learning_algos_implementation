import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize

# Define transformations at the top level of the module
transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,)) # Normalize for grayscale images
])

# Define the transform application function at the top level
def apply_transforms(batch):
    batch['image'] = [transforms(img) for img in batch['image']]
    return batch

def prepare_data(num_clients: int, non_iid_alpha: float, test_split: float = 0.2):
    """
    Ensures the FEMNIST dataset is downloaded, split, and partitioned. It saves the
    splits and client data indices to disk for efficient loading by parallel workers.

    Args:
        num_clients (int): The total number of clients.
        non_iid_alpha (float): The concentration parameter for the Dirichlet distribution.
                               A smaller alpha creates a more skewed, non-IID distribution.
        test_split (float): The fraction of data to use for the test set.

    Returns:
        tuple: A tuple containing:
            - str: Path to the cached training dataset.
            - str: Path to the cached test dataset.
            - list[list[int]]: The indices of the training data assigned to each client.
    """
    # Define path for cached splits
    splits_path = os.path.join(os.path.dirname(__file__), "..", "data", "femnist_splits")

    # 2. Load or create the dataset splits
    if os.path.exists(splits_path):
        print(f"Pre-split dataset found at {splits_path}. Skipping download and split.")
    else:
        print("Loading FEMNIST dataset from Hugging Face...")
        dataset = load_dataset("flwrlabs/femnist")
        
        # This is a one-time operation
        # The dataset only has a 'train' split. We create our own test set.
        print(f"Splitting dataset into train and test sets ({1-test_split:.0%}/{test_split:.0%})...")
        train_test_split = dataset['train'].train_test_split(test_size=test_split, seed=42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        print(f"Saving splits to {splits_path}...")
        os.makedirs(splits_path, exist_ok=True)
        train_dataset.save_to_disk(os.path.join(splits_path, "train"))
        test_dataset.save_to_disk(os.path.join(splits_path, "test"))

    # This logic runs every time to generate the client data distribution
    train_dataset_path = os.path.join(splits_path, "train")
    train_dataset = load_from_disk(train_dataset_path)

    num_classes = len(train_dataset.features['character'].names)

    print(f"Distributing data among {num_clients} clients with non-IID alpha={non_iid_alpha}...")
    
    # 3. Partition the training data using a Dirichlet distribution for non-IID split
    # Get all labels from the training dataset
    labels = np.array(train_dataset['character'])
    
    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute its data samples among clients based on Dirichlet
    for k_indices in class_indices:
        # Ensure there are samples to distribute for the class
        if len(k_indices) == 0:
            continue
            
        # Shuffle indices for randomness
        np.random.shuffle(k_indices)
        
        # Generate proportions for the current class for all clients
        proportions = np.random.dirichlet(np.repeat(non_iid_alpha, num_clients))
        
        # Split the indices for the current class and assign to clients
        # The last client gets all remaining indices
        split_points = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
        client_class_splits = np.split(k_indices, split_points)

        for i in range(num_clients):
            # Ensure that we do not go out of bounds if a client gets no data for a class
            if i < len(client_class_splits):
                client_indices[i].extend(client_class_splits[i])
    
    test_dataset_path = os.path.join(splits_path, "test")
    
    return train_dataset_path, test_dataset_path, client_indices

if __name__ == '__main__':
    # Example usage of the function
    num_classes = 62
    NUM_CLIENTS = 10
    # A smaller alpha (e.g., 0.5) creates a more skewed distribution.
    # A larger alpha (e.g., 100) creates a more uniform distribution.
    ALPHA = 0.5
    BATCH_SIZE = 32
    
    train_path, test_path, client_indices = prepare_data(
        num_clients=NUM_CLIENTS,
        non_iid_alpha=ALPHA,
    )
    
    print(f"\nData prepared. Train path: {train_path}, Test path: {test_path}")
    print(f"Generated data indices for {len(client_indices)} clients.")

    # To inspect, you would now load the data and create a loader
    train_dataset = load_from_disk(train_path).with_transform(apply_transforms)
    first_client_subset = Subset(train_dataset, client_indices[0])
    first_client_loader = DataLoader(first_client_subset, batch_size=BATCH_SIZE)
    
    first_batch = next(iter(first_client_loader))
    images, labels = first_batch['image'], first_batch['character']
    
    print(f"\nShape of a batch of images for client 0: {images.shape}") # Should be [BATCH_SIZE, 1, 28, 28]
    print(f"Shape of a batch of labels for client 0: {labels.shape}") # Should be [BATCH_SIZE]

    # Inspect the data distribution for each client
    print("\n--- Data Distribution Inspection ---")
    all_client_distributions = []
    for i, indices in enumerate(client_indices):
        # Get the labels for these specific indices from the main training dataset
        client_labels = [train_dataset[j]['character'] for j in indices]
        
        label_counts = {label: client_labels.count(label) for label in range(num_classes)}
        all_client_distributions.append(label_counts)
        
        print(f"\nClient {i} distribution:")
        # Print non-zero counts for brevity
        for label, count in sorted(label_counts.items()):
            if count > 0:
                print(f"  Class {label}: {count} samples")

    # Optional: Plot the distribution for a visual representation
    try:
        import matplotlib.pyplot as plt
        # You can add plotting code here to visualize the distributions.
        # For example, plot the distribution for the first client.
        output_folder = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(output_folder, exist_ok=True)
        for i in range(NUM_CLIENTS):      
            plot_path = os.path.join(output_folder, f"client_{i}_distribution.png")

            first_client_dist = all_client_distributions[i]
            plt.figure(figsize=(15, 5))
            plt.bar(first_client_dist.keys(), first_client_dist.values())
            plt.title(f"Data Distribution for Client {i}")
            plt.xlabel("Class Label")
            plt.ylabel("Number of Samples")
            plt.savefig(plot_path)
            print(f"\nSaved client {i} distribution plot to: {plot_path}")
    except ImportError:
        print("\nMatplotlib not found. Skipping plot. Install with: pip install matplotlib")
