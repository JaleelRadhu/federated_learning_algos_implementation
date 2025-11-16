import argparse
import os
import random
import json
import time
import multiprocessing as mp
import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset

from data.data_utils import prepare_data, apply_transforms
from model import CNN
# Import your algorithm-specific classes
from optimizers.fedadam import FedAdamClient
from optimizers.fedadam_v2 import FedAdamV2Client, FedAdamV2Server
from optimizers.fedrmsprop import FedRMSPropClient, FedRMSPropServer
from optimizers.fedadagrad import FedAdagradClient, FedAdagradServer
from optimizers.fedamsgrad import FedAMSGradClient, FedAMSGradServer
from optimizers.fedadamw import FedAdamWClient, FedAdamWServer
from optimizers.fedavg import FedAvgClient, FedAvgServer

# --- Algorithm Registration ---
# A mapping from algorithm names to their corresponding client and server classes.
ALGORITHMS = {
    'fedavg': {'client': FedAvgClient, 'server': FedAvgServer},
    'fedadam': {'client': FedAdamClient, 'server': FedAvgServer}, # Original client-side Adam
    'fedadam_v2': {'client': FedAdamV2Client, 'server': FedAdamV2Server}, # Method 2: Server-side Adam
    'fedrmsprop': {'client': FedRMSPropClient, 'server': FedRMSPropServer}, # Server-side RMSProp
    'fedadagrad': {'client': FedAdagradClient, 'server': FedAdagradServer}, # Server-side Adagrad
    'fedamsgrad': {'client': FedAMSGradClient, 'server': FedAMSGradServer}, # Server-side AMSGrad
    'fedadamw': {'client': FedAdamWClient, 'server': FedAdamWServer}, # Server-side AdamW
}

# --- Worker Initialization ---
# We define a global variable that will hold the dataset for each worker process.
# This avoids loading the dataset from disk for every single task.
worker_train_dataset = None

def init_worker(train_dataset_path):
    """
    Initializer for each worker process in the pool.
    Loads the dataset once per process.
    """
    global worker_train_dataset
    # Load the dataset from disk and apply transforms. This is done once per worker.
    worker_train_dataset = load_from_disk(train_dataset_path).with_transform(apply_transforms)

def train_client_worker(worker_args):
    """
    A worker function that can be executed in parallel by a multiprocessing pool.
    It trains a single client.

    Args:
        worker_args (tuple): A tuple containing the arguments for training a client:
            - client_id (int): The ID of the client.
            - global_model (nn.Module): The shared global model object.
            - client_indices (list[int]): The indices for this client's data subset.
            - args (Namespace): The command-line arguments.
            - device (torch.device): The device to train on (e.g., 'cuda:0').
    Returns:
        tuple: A tuple containing the results from the client.
            - client_id (int): The ID of the client.
            - state_dict (OrderedDict): The updated model state_dict.
            - num_samples (int): The number of samples in the client's dataset.
            - loss (float): The average training loss.
    """
    client_id, global_model, client_indices, args, device = worker_args
    global worker_train_dataset

    # Create a DataLoader for this specific client
    client_subset = Subset(worker_train_dataset, client_indices)
    dataloader = DataLoader(
        client_subset,
        batch_size=args.batch_size,
        shuffle=True,
        # Set num_workers to 0 inside the worker process.
        # A DataLoader with num_workers > 0 cannot be used inside a daemonic process,
        # which is what multiprocessing.Pool uses.
        num_workers=0,
        pin_memory=True
    )
    
    # Create a new local model instance for this client.
    # This is much faster than deep-copying the shared model.
    local_model = CNN(num_classes=62)
    # Load the weights from the shared global model.
    local_model.load_state_dict(global_model.state_dict())

    # Dynamically select the client class based on the algorithm name
    algo_name = args.algo.lower()
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Algorithm {args.algo} not supported in worker.")
    
    ClientClass = ALGORITHMS[algo_name]['client']

    # Instantiate the client for this specific process
    client = ClientClass(
        client_id=client_id,
        model=local_model,  # Pass the new local model
        dataloader=dataloader,
        learning_rate=args.lr, device=device
    )
    loss = client.train(local_epochs=args.local_epochs)
    
    # Move state_dict to CPU before putting it in the queue
    cpu_state_dict = {k: v.cpu() for k, v in client.model.state_dict().items()}
    return (client_id, cpu_state_dict, len(dataloader.dataset), loss)


def main(args):
    """
    The main function to run the federated learning simulation.
    """
    # 1. Setup
    # =================================================================================
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"--- Main process on device: {device} ---")

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Create results directory
    results_path = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_path, exist_ok=True)

    # Create a unique run name for saving results
    run_name = (
        f"{args.algo}_rounds{args.rounds}_clients{args.num_clients}_frac{args.client_fraction}"
        f"_epochs{args.local_epochs}_lr{args.lr}_bs{args.batch_size}_alpha{args.alpha}"
    )
    print(f"\nRun Name: {run_name}")

    # 2. Data and Model
    # =================================================================================
    start_time = time.time()
    train_dataset_path, test_dataset_path, client_indices_list = prepare_data(
        num_clients=args.num_clients,
        non_iid_alpha=args.alpha,
        test_split=0.2
    )
    end_time = time.time()
    loading_duration = (end_time - start_time)
    print(f"--- Data Loading Duration: {loading_duration:.2f} seconds ---")

    
    # Instantiate the global model
    test_dataset = load_from_disk(test_dataset_path).with_transform(apply_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    # Keep the global model on the CPU. It will be moved to the correct
    # device inside the server and client worker processes.
    global_model = CNN(num_classes=62)

    # Move model to shared memory. This is crucial for efficient multiprocessing.
    global_model.share_memory()

    # 3. Initialize Server and Clients
    # =================================================================================
    # Dynamically select the server class based on the algorithm name
    algo_name = args.algo.lower()
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Algorithm {args.algo} is not supported.")
    
    ServerClass = ALGORITHMS[algo_name]['server']
    server = ServerClass(
        model=global_model,
        test_loader=test_loader,
        learning_rate=args.lr,
        device=device
    )

    # 4. Federated Learning Loop
    # =================================================================================
    print("\n--- Starting Federated Learning Simulation ---")
    
    train_losses = []
    test_accuracies = []
    test_losses = []

    for round_num in (round_iterator := trange(args.rounds, desc="Federated Rounds")):
        # --- Client Selection ---
        num_selected_clients = max(int(args.num_clients * args.client_fraction), 1)
        selected_client_ids = random.sample(range(args.num_clients), num_selected_clients)
        
        # --- Client Training ---
        # Prepare arguments for the worker processes
        worker_args = []
        for i, client_id in enumerate(selected_client_ids):
            # Distribute clients across available GPUs
            gpu_id = i % num_gpus if num_gpus > 0 else -1
            worker_device = torch.device(f"cuda:{gpu_id}") if gpu_id != -1 else torch.device("cpu")
            
            worker_args.append((
                client_id,
                server.model,  # Pass the shared model object directly
                client_indices_list[client_id],
                args,
                worker_device
            ))

        # Use a multiprocessing pool to manage worker processes
        # This is more resource-efficient than spawning new processes every round
        # Use a number of processes up to the number of selected clients or available CPUs
        num_processes = min(num_selected_clients, mp.cpu_count(), 16)
        
        # The initializer loads the dataset into each worker process once.
        # This drastically reduces the overhead per task.
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(train_dataset_path,)) as pool:
            # Use tqdm to create a progress bar for the client training
            results = list(tqdm(pool.imap(train_client_worker, worker_args), 
                                total=num_selected_clients, 
                                desc=f"Round {round_num+1} Client Training", leave=False))
            
            # Explicitly close and join the pool to ensure graceful shutdown of workers
            pool.close()
            pool.join()

        # Unpack and sort results by client_id for consistency
        results.sort(key=lambda x: x[0])
        _, client_state_dicts, client_weights, client_losses = zip(*results)
        round_client_loss = sum(client_losses)

        avg_client_loss = round_client_loss / num_selected_clients
        train_losses.append(avg_client_loss)

        # --- Server Aggregation ---
        server.update_model(client_state_dicts, client_weights)

        # --- Server Evaluation ---
        test_loss, test_accuracy = server.test()
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Update the round progress bar description
        round_iterator.set_description(f"Round {round_num+1} | Acc: {test_accuracy:.2f}%")

    # 5. Plotting Results
    # =================================================================================
    print("\n--- Simulation Finished ---")
    
    # Plot Training and Test Loss vs. Round
    plt.figure()
    plt.plot(range(1, args.rounds + 1), train_losses, label='Average Client Training Loss')
    plt.plot(range(1, args.rounds + 1), test_losses, label='Global Model Test Loss')
    plt.title("Training and Test Loss vs. Communication Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(results_path, f"{run_name}_loss_comparison.png")
    plt.savefig(loss_plot_path)
    print(f"Saved loss comparison plot to: {loss_plot_path}")

    # Plot Test Accuracy vs. Round
    plt.figure()
    plt.plot(range(1, args.rounds + 1), test_accuracies)
    plt.title("Global Model Test Accuracy vs. Communication Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    acc_plot_path = os.path.join(results_path, f"{run_name}_test_accuracy.png")
    plt.savefig(acc_plot_path)
    print(f"Saved test accuracy plot to: {acc_plot_path}")

    # 6. Save Results to JSON
    # =================================================================================
    results_data = {
        "hyperparameters": vars(args),
        "results": {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
        }
    }
    json_path = os.path.join(results_path, f"{run_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved detailed results to: {json_path}")


if __name__ == '__main__':
    # Set the start method for multiprocessing
    # 'spawn' is recommended for CUDA compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Federated Learning Simulation Framework")
    
    # FL Parameters
    parser.add_argument('--algo', type=str, default='fedavg', choices=sorted(ALGORITHMS.keys()), help="Federated learning algorithm")
    parser.add_argument('--rounds', type=int, default=10, help="Number of communication rounds")
    parser.add_argument('--num_clients', type=int, default=100, help="Total number of clients")
    parser.add_argument('--client_fraction', type=float, default=0.1, help="Fraction of clients to select each round")
    
    # Model and Training Parameters
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the client optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the dataloaders")
    parser.add_argument('--local_epochs', type=int, default=5, help="Number of local epochs for client training")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
    
    # Data Parameters
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha for Dirichlet distribution (non-IID)")

    args = parser.parse_args()
    main(args)