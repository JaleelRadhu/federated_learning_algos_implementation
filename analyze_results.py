import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(results_dir='results'):
    """Load all result JSON files from the results directory"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
    return results

def parse_results(results):
    """Parse and organize results by different parameters"""
    organized = {
        'by_algo': defaultdict(list),
        'by_lr': defaultdict(list),
        'by_epochs': defaultdict(list),
        'by_batch_size': defaultdict(list),
        'by_alpha': defaultdict(list),
    }
    
    for result in results:
        hp = result['hyperparameters']
        algo = hp['algo']
        lr = hp['lr']
        epochs = hp['local_epochs']
        bs = hp['batch_size']
        alpha = hp['alpha']
        
        # Store result with all parameters for filtering
        result_entry = {
            'hyperparameters': hp,
            'results': result['results'],
            'final_accuracy': result['results']['test_accuracies'][-1],
            'final_loss': result['results']['test_losses'][-1],
        }
        
        organized['by_algo'][algo].append(result_entry)
        organized['by_lr'][lr].append(result_entry)
        organized['by_epochs'][epochs].append(result_entry)
        organized['by_batch_size'][bs].append(result_entry)
        organized['by_alpha'][alpha].append(result_entry)
    
    return organized

def plot_algorithm_comparison(results, save_dir='analysis_plots'):
    """Compare different algorithms with fixed hyperparameters"""
    comparison_dir = os.path.join(save_dir, '01_algorithm_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Filter for fair comparison: lr=0.001, epochs=1, bs=32, alpha=0.5
    filtered_results = {}
    for result in results:
        hp = result['hyperparameters']
        if hp['lr'] == 0.001 and hp['local_epochs'] == 1 and hp['batch_size'] == 32 and hp['alpha'] == 0.5:
            filtered_results[hp['algo']] = result['results']
    
    if not filtered_results:
        print("No results found for algorithm comparison with specified parameters")
        return
    
    # Plot 1: Test Accuracy Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for algo, res in filtered_results.items():
        rounds = list(range(1, len(res['test_accuracies']) + 1))
        ax1.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Comparison: Test Accuracy\n(lr=0.001, epochs=1, bs=32, alpha=0.5)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Loss Comparison
    for algo, res in filtered_results.items():
        rounds = list(range(1, len(res['test_losses']) + 1))
        ax2.plot(rounds, res['test_losses'], marker='o', label=algo, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Comparison: Test Loss\n(lr=0.001, epochs=1, bs=32, alpha=0.5)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {comparison_dir}/algorithm_comparison.png")
    plt.close()

def plot_learning_rate_impact(results, save_dir='analysis_plots'):
    """Analyze impact of learning rate - combined plot for all algorithms"""
    lr_dir = os.path.join(save_dir, '02_learning_rate_impact')
    os.makedirs(lr_dir, exist_ok=True)
    
    # Get unique algorithms and learning rates
    algos = sorted(set(r['hyperparameters']['algo'] for r in results))
    learning_rates = sorted(set(r['hyperparameters']['lr'] for r in results))
    
    # Create subplots: one row per learning rate
    fig, axes = plt.subplots(len(learning_rates), 2, figsize=(16, 5 * len(learning_rates)))
    if len(learning_rates) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, lr in enumerate(learning_rates):
        ax1, ax2 = axes[idx]
        
        for algo in algos:
            # Filter for this algorithm with specific lr, epochs=1, bs=32, alpha=0.5
            for result in results:
                hp = result['hyperparameters']
                if (hp['algo'] == algo and hp['lr'] == lr and hp['local_epochs'] == 1 and 
                    hp['batch_size'] == 32 and hp['alpha'] == 0.5):
                    res = result['results']
                    rounds = list(range(1, len(res['test_accuracies']) + 1))
                    ax1.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2, markersize=3)
                    ax2.plot(rounds, res['test_losses'], marker='o', label=algo, linewidth=2, markersize=3)
                    break
        
        ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Learning Rate = {lr}: Test Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
        ax2.set_title(f'Learning Rate = {lr}: Test Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Rate Impact Comparison (epochs=1, bs=32, alpha=0.5)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(lr_dir, 'learning_rate_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {lr_dir}/learning_rate_comparison.png")
    plt.close()

def plot_local_epochs_impact(results, save_dir='analysis_plots'):
    """Analyze impact of local epochs - combined plot for all algorithms"""
    epochs_dir = os.path.join(save_dir, '03_local_epochs_impact')
    os.makedirs(epochs_dir, exist_ok=True)
    
    # Get unique algorithms and local epochs values
    algos = sorted(set(r['hyperparameters']['algo'] for r in results))
    local_epochs = sorted(set(r['hyperparameters']['local_epochs'] for r in results))
    
    # Create subplots: one row per epochs value
    fig, axes = plt.subplots(len(local_epochs), 2, figsize=(16, 5 * len(local_epochs)))
    if len(local_epochs) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, epochs in enumerate(local_epochs):
        ax1, ax2 = axes[idx]
        
        for algo in algos:
            # Filter for specific configuration
            for result in results:
                hp = result['hyperparameters']
                if (hp['algo'] == algo and hp['local_epochs'] == epochs and 
                    hp['lr'] == 0.001 and hp['batch_size'] == 32 and hp['alpha'] == 0.5):
                    res = result['results']
                    rounds = list(range(1, len(res['test_accuracies']) + 1))
                    ax1.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2, markersize=3)
                    ax2.plot(rounds, res['test_losses'], marker='o', label=algo, linewidth=2, markersize=3)
                    break
        
        ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Local Epochs = {epochs}: Test Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
        ax2.set_title(f'Local Epochs = {epochs}: Test Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Local Epochs Impact Comparison (lr=0.001, bs=32, alpha=0.5)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(epochs_dir, 'local_epochs_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {epochs_dir}/local_epochs_comparison.png")
    plt.close()

def plot_batch_size_impact(results, save_dir='analysis_plots'):
    """Analyze impact of batch size - combined plot for all algorithms"""
    bs_dir = os.path.join(save_dir, '04_batch_size_impact')
    os.makedirs(bs_dir, exist_ok=True)
    
    # Get unique algorithms and batch sizes
    algos = sorted(set(r['hyperparameters']['algo'] for r in results))
    batch_sizes = sorted(set(r['hyperparameters']['batch_size'] for r in results))
    
    # Single figure with subplots for each batch size
    fig, axes = plt.subplots(len(batch_sizes), 2, figsize=(16, 5 * len(batch_sizes)))
    if len(batch_sizes) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, bs in enumerate(batch_sizes):
        ax1, ax2 = axes[idx] if len(batch_sizes) > 1 else axes
        
        for algo in algos:
            # Filter for specific configuration
            for result in results:
                hp = result['hyperparameters']
                if (hp['algo'] == algo and hp['batch_size'] == bs and 
                    hp['lr'] == 0.001 and hp['local_epochs'] == 1 and hp['alpha'] == 0.5):
                    res = result['results']
                    rounds = list(range(1, len(res['test_accuracies']) + 1))
                    ax1.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2, markersize=3)
                    ax2.plot(rounds, res['test_losses'], marker='o', label=algo, linewidth=2, markersize=3)
                    break
        
        ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Batch Size = {bs}: Test Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
        ax2.set_title(f'Batch Size = {bs}: Test Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Batch Size Impact Comparison (lr=0.001, epochs=1, alpha=0.5)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(bs_dir, 'batch_size_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {bs_dir}/batch_size_comparison.png")
    plt.close()

def plot_alpha_impact(results, save_dir='analysis_plots'):
    """Analyze impact of alpha (data heterogeneity) - combined plot for all algorithms"""
    alpha_dir = os.path.join(save_dir, '05_alpha_impact')
    os.makedirs(alpha_dir, exist_ok=True)
    
    # Get unique algorithms and alpha values
    algos = sorted(set(r['hyperparameters']['algo'] for r in results))
    alpha_values = sorted(set(r['hyperparameters']['alpha'] for r in results))
    
    # Create subplots: one row per alpha value
    fig, axes = plt.subplots(len(alpha_values), 2, figsize=(16, 5 * len(alpha_values)))
    if len(alpha_values) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, alpha in enumerate(alpha_values):
        ax1, ax2 = axes[idx] if len(alpha_values) > 1 else axes
        
        for algo in algos:
            # Filter for specific configuration
            for result in results:
                hp = result['hyperparameters']
                if (hp['algo'] == algo and hp['alpha'] == alpha and 
                    hp['lr'] == 0.001 and hp['local_epochs'] == 1 and hp['batch_size'] == 32):
                    res = result['results']
                    rounds = list(range(1, len(res['test_accuracies']) + 1))
                    ax1.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2, markersize=3)
                    ax2.plot(rounds, res['test_losses'], marker='o', label=algo, linewidth=2, markersize=3)
                    break
        
        ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Alpha = {alpha}: Test Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
        ax2.set_title(f'Alpha = {alpha}: Test Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Data Heterogeneity (Alpha) Impact Comparison (lr=0.001, epochs=1, bs=32)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(alpha_dir, 'alpha_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {alpha_dir}/alpha_comparison.png")
    plt.close()

def create_summary_table(results, save_dir='analysis_plots'):
    """Create a summary table of best configurations for each algorithm"""
    summary_dir = os.path.join(save_dir, '00_summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Group by algorithm and find best configuration
    algo_best = {}
    for result in results:
        hp = result['hyperparameters']
        algo = hp['algo']
        final_acc = result['results']['test_accuracies'][-1]
        
        if algo not in algo_best or final_acc > algo_best[algo]['accuracy']:
            algo_best[algo] = {
                'accuracy': final_acc,
                'loss': result['results']['test_losses'][-1],
                'lr': hp['lr'],
                'epochs': hp['local_epochs'],
                'batch_size': hp['batch_size'],
                'alpha': hp['alpha']
            }
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Algorithm', 'Best Accuracy (%)', 'Final Loss', 'Learning Rate', 'Local Epochs', 'Batch Size', 'Alpha']
    table_data = []
    
    for algo in sorted(algo_best.keys()):
        best = algo_best[algo]
        table_data.append([
            algo.upper(),
            f"{best['accuracy']:.2f}",
            f"{best['loss']:.4f}",
            f"{best['lr']}",
            f"{best['epochs']}",
            f"{best['batch_size']}",
            f"{best['alpha']}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.12, 0.14, 0.14, 0.12, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Best Configuration for Each Algorithm', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(summary_dir, 'best_configurations_summary.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {summary_dir}/best_configurations_summary.png")
    plt.close()
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY: Best Configuration for Each Algorithm")
    print("="*80)
    for algo in sorted(algo_best.keys()):
        best = algo_best[algo]
        print(f"\n{algo.upper()}:")
        print(f"  Best Accuracy: {best['accuracy']:.2f}%")
        print(f"  Final Loss: {best['loss']:.4f}")
        print(f"  Hyperparameters: lr={best['lr']}, epochs={best['epochs']}, bs={best['batch_size']}, alpha={best['alpha']}")
    print("="*80 + "\n")

def plot_convergence_comparison(results, save_dir='analysis_plots'):
    """Compare convergence speed of different algorithms"""
    comparison_dir = os.path.join(save_dir, '01_algorithm_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Filter for fair comparison
    filtered = {}
    for result in results:
        hp = result['hyperparameters']
        if hp['lr'] == 0.001 and hp['local_epochs'] == 1 and hp['batch_size'] == 32 and hp['alpha'] == 0.5:
            algo = hp['algo']
            filtered[algo] = result['results']
    
    if len(filtered) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot accuracy curves
    for algo, res in filtered.items():
        rounds = list(range(1, len(res['test_accuracies']) + 1))
        ax.plot(rounds, res['test_accuracies'], marker='o', label=algo, linewidth=2.5, markersize=5)
    
    ax.set_xlabel('Communication Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Convergence Comparison: All Algorithms\n(lr=0.001, epochs=1, bs=32, alpha=0.5)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {comparison_dir}/convergence_comparison.png")
    plt.close()

def main():
    print("Loading results from 'results' directory...")
    results = load_results('results')
    print(f"Loaded {len(results)} result files\n")
    
    print("Generating analysis plots...")
    print("-" * 80)
    
    # Generate all plots
    plot_algorithm_comparison(results)
    plot_learning_rate_impact(results)
    plot_local_epochs_impact(results)
    plot_batch_size_impact(results)
    plot_alpha_impact(results)
    plot_convergence_comparison(results)
    create_summary_table(results)
    
    print("-" * 80)
    print("\nAnalysis complete! All plots saved in 'analysis_plots' directory.")
    print(f"Total plots generated: Check the 'analysis_plots' folder")

if __name__ == "__main__":
    main()
