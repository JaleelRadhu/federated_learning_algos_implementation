import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")

def create_combined_client_distribution(data_dir='data', save_dir='analysis_plots/00_summary'):
    """Create a combined figure showing all client distributions from existing images"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_clients = 10
    
    # Create a grid layout (2 rows x 5 columns for 10 clients)
    fig = plt.figure(figsize=(22, 7))
    
    for i in range(num_clients):
        ax = plt.subplot(2, 5, i + 1)
        img_path = os.path.join(data_dir, f'client_{i}_distribution.png')
        
        if os.path.exists(img_path):
            # Load and display the image
            img = mpimg.imread(img_path)
            ax.imshow(img, aspect='auto')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Image not found:\n{img_path}', 
                        ha='center', va='center', fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Data Distribution Across All Clients (Dirichlet Distribution - Non-IID)', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02, hspace=0.08)
    
    output_path = os.path.join(save_dir, 'all_clients_distribution_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()

def create_selected_clients_view(data_dir='data', save_dir='analysis_plots/00_summary', clients=[0, 1, 2]):
    """Create a figure showing selected client distributions for main paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_selected = len(clients)
    
    # Create horizontal layout
    fig = plt.figure(figsize=(5.5 * num_selected, 4))
    
    for idx, client_id in enumerate(clients):
        ax = plt.subplot(1, num_selected, idx + 1)
        img_path = os.path.join(data_dir, f'client_{client_id}_distribution.png')
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img, aspect='auto')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Image not found', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    plt.suptitle('Representative Client Data Distributions (Non-IID with Dirichlet)', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03)
    
    output_path = os.path.join(save_dir, 'selected_clients_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()

def create_grid_view(data_dir='data', save_dir='analysis_plots/00_summary', rows=2, cols=3):
    """Create a compact grid view of first N clients for paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_clients = rows * cols
    
    fig = plt.figure(figsize=(4.5 * cols, 3.2 * rows))
    
    for i in range(num_clients):
        ax = plt.subplot(rows, cols, i + 1)
        img_path = os.path.join(data_dir, f'client_{i}_distribution.png')
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img, aspect='auto')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Client {i}\nImage not found', 
                        ha='center', va='center', fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Client Data Distribution Examples (Dirichlet Non-IID)', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.02, hspace=0.06)
    
    output_path = os.path.join(save_dir, f'clients_grid_{rows}x{cols}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("Creating Combined Data Distribution Visualizations from Existing Images")
    print("="*80)
    
    data_dir = 'data'
    save_dir = 'analysis_plots/00_summary'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        return
    
    # Check for existing distribution images
    existing_images = [f for f in os.listdir(data_dir) if f.endswith('_distribution.png')]
    print(f"\nFound {len(existing_images)} client distribution images in '{data_dir}'")
    
    if len(existing_images) == 0:
        print("No distribution images found. Please run data_utils.py first to generate them.")
        return
    
    # Create different visualization options
    print("\n1. Creating combined view of all 10 clients...")
    create_combined_client_distribution(data_dir, save_dir)
    
    print("\n2. Creating selected clients view (3 clients for main paper)...")
    create_selected_clients_view(data_dir, save_dir, clients=[0, 1, 2])
    
    print("\n3. Creating compact grid view (2x3)...")
    create_grid_view(data_dir, save_dir, rows=2, cols=3)
    
    print("\n4. Creating compact grid view (3x3)...")
    create_grid_view(data_dir, save_dir, rows=3, cols=3)
    
    print("\n" + "="*80)
    print("Data visualization generation complete!")
    print(f"All plots saved in '{save_dir}' directory")
    print("="*80)
    print("\nRecommendations for paper:")
    print("  - Use 'selected_clients_distribution.png' in main paper (3 clients)")
    print("  - Use 'clients_grid_2x3.png' or 'clients_grid_3x3.png' for supplementary")
    print("  - Use 'all_clients_distribution_combined.png' in appendix (all 10 clients)")
    print("="*80)

if __name__ == "__main__":
    main()
