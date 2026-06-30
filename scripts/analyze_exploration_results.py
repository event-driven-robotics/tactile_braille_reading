"""analyze_exploration_results.py

Analyze and visualize results from the hidden neuron exploration experiment.

This script loads training results from multiple network configurations and creates
comparative visualizations showing how performance scales with network size.

Usage:
    python scripts/analyze_exploration_results.py --exploration-dir ./results/20260114_1030_exploration/

Options:
    --exploration-dir PATH    Path to exploration results directory
    --output-dir PATH         Output directory for analysis plots (default: ./figures)
    --title TEXT              Custom title for plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_exploration_results(exploration_dir):
    """Load all results files from exploration directory.
    
    Parameters
    ----------
    exploration_dir : str
        Path to exploration results directory. Can contain timestamped subdirectories.
        
    Returns
    -------
    dict
        Dictionary mapping neuron counts to results dictionaries
    """
    exploration_path = Path(exploration_dir)
    if not exploration_path.exists():
        raise FileNotFoundError(f"Directory not found: {exploration_dir}")
    
    results = {}
    
    # Find all .npz files (including in timestamped subdirectories)
    for npz_file in exploration_path.rglob("*.npz"):
        try:
            data = np.load(npz_file)
            
            # Try to extract neuron count from filename
            # Format: braille_reading_rsnn_NN_neurons_...
            filename = npz_file.stem
            parts = filename.split("_")
            nb_hidden_idx = parts.index("neurons") - 1
            nb_hidden = int(parts[nb_hidden_idx])
            
            results[nb_hidden] = {
                'filename': str(npz_file),
                'acc_train': data['acc_train'],
                'acc_test': data['acc_test'],
                'loss_train': data['loss_train'],
                'val_acc': float(data['val_acc']) if 'val_acc' in data else None,
            }
            print(f"Loaded {nb_hidden} neurons: {npz_file.name}")
        except Exception as e:
            print(f"Warning: Could not load {npz_file.name}: {e}")
    
    return dict(sorted(results.items()))


def plot_performance_scaling(results, output_dir, title=""):
    """Create plots showing performance scaling with network size.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping neuron counts to results
    output_dir : str
        Output directory for plots
    title : str
        Custom title (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    neuron_counts = sorted(results.keys())
    
    # Extract metrics
    final_train_accs = []
    final_test_accs = []
    best_train_accs = []
    best_test_accs = []
    val_accs = []
    
    for nb in neuron_counts:
        data = results[nb]
        
        # Final accuracies
        final_train_accs.append(data['acc_train'][-1])
        final_test_accs.append(data['acc_test'][-1])
        
        # Best accuracies
        best_train_accs.append(np.max(data['acc_train']))
        best_test_accs.append(np.max(data['acc_test']))
        
        # Validation accuracy
        if data['val_acc'] is not None:
            val_accs.append(data['val_acc'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Final vs Best Accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(neuron_counts, final_train_accs, 'o-', label='Final Training', linewidth=2, markersize=8)
    ax1.plot(neuron_counts, final_test_accs, 's-', label='Final Test', linewidth=2, markersize=8)
    ax1.plot(neuron_counts, best_train_accs, 'o--', label='Best Training', alpha=0.6, linewidth=2)
    ax1.plot(neuron_counts, best_test_accs, 's--', label='Best Test', alpha=0.6, linewidth=2)
    if val_accs:
        ax1.plot(neuron_counts, val_accs, '^-', label='Validation', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Hidden Neurons')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Network Size')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.05)
    
    # Plot 2: Test Accuracy Improvement
    ax2 = fig.add_subplot(2, 2, 2)
    test_improvement = np.array(best_test_accs) - np.array(final_test_accs)
    ax2.bar(range(len(neuron_counts)), test_improvement, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(neuron_counts)))
    ax2.set_xticklabels(neuron_counts)
    ax2.set_xlabel('Number of Hidden Neurons')
    ax2.set_ylabel('Improvement (Best - Final)')
    ax2.set_title('Training Improvement per Configuration')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Learning Curves for each config
    ax3 = fig.add_subplot(2, 2, 3)
    for nb in neuron_counts:
        data = results[nb]
        epochs = range(1, len(data['acc_test']) + 1)
        ax3.plot(epochs, data['acc_test'], label=f'{nb} neurons', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Test Accuracy Learning Curves')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.0, 1.05)
    
    # Plot 4: Final Accuracy Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(len(neuron_counts))
    width = 0.35
    ax4.bar(x - width/2, final_train_accs, width, label='Training', alpha=0.8)
    ax4.bar(x + width/2, final_test_accs, width, label='Test', alpha=0.8)
    ax4.set_xlabel('Number of Hidden Neurons')
    ax4.set_ylabel('Final Accuracy')
    ax4.set_title('Final Accuracy Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(neuron_counts)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0.0, 1.05)
    
    # Overall title
    overall_title = title or "Hidden Neuron Exploration Results"
    fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    output_file = os.path.join(output_dir, 'exploration_analysis.pdf')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {output_file}")
    plt.close(fig)
    
    # Print summary table
    print("\n" + "="*70)
    print("EXPLORATION SUMMARY")
    print("="*70)
    print(f"{'Neurons':<10} {'Final Train':<15} {'Final Test':<15} {'Best Test':<15} {'Improvement':<15}")
    print("-"*70)
    for i, nb in enumerate(neuron_counts):
        improvement = test_improvement[i]
        print(f"{nb:<10} {final_train_accs[i]:.4f} ({100*final_train_accs[i]:.2f}%) "
              f"{final_test_accs[i]:.4f} ({100*final_test_accs[i]:.2f}%) "
              f"{best_test_accs[i]:.4f} ({100*best_test_accs[i]:.2f}%) "
              f"{improvement:.4f} ({100*improvement:.2f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hidden neuron exploration results'
    )
    parser.add_argument(
        '--exploration-dir',
        required=True,
        help='Path to exploration results directory'
    )
    parser.add_argument(
        '--output-dir',
        default='./figures',
        help='Output directory for analysis plots'
    )
    parser.add_argument(
        '--title',
        default='',
        help='Custom title for plots'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.exploration_dir}")
    results = load_exploration_results(args.exploration_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(results)} configurations")
    plot_performance_scaling(results, args.output_dir, args.title)


if __name__ == '__main__':
    main()
