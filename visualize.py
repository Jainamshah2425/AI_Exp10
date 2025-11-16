"""
Visualization Script
Creates comparison plots of training curves, accuracy, and loss
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results():
    """Load training results from pickle file"""
    results_path = os.path.join(os.path.dirname(__file__), 'results', 'training_results.pkl')
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("   Please run 'python train.py' first to generate results.")
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def plot_training_comparison(results):
    """Create comprehensive training comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Comparison: Flawed vs Improved Model\nBased on Glorot & Bengio (2010)', 
                 fontsize=16, fontweight='bold')
    
    flawed = results['flawed']['history']
    improved = results['improved']['history']
    epochs = range(1, len(flawed['loss']) + 1)
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, flawed['loss'], 'r-', linewidth=2, label='Flawed Model', marker='o')
    ax1.plot(epochs, improved['loss'], 'g-', linewidth=2, label='Improved Model', marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, flawed['val_loss'], 'r-', linewidth=2, label='Flawed Model', marker='o')
    ax2.plot(epochs, improved['val_loss'], 'g-', linewidth=2, label='Improved Model', marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, [acc*100 for acc in flawed['accuracy']], 'r-', 
             linewidth=2, label='Flawed Model', marker='o')
    ax3.plot(epochs, [acc*100 for acc in improved['accuracy']], 'g-', 
             linewidth=2, label='Improved Model', marker='s')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 1]
    ax4.plot(epochs, [acc*100 for acc in flawed['val_accuracy']], 'r-', 
             linewidth=2, label='Flawed Model', marker='o')
    ax4.plot(epochs, [acc*100 for acc in improved['val_accuracy']], 'g-', 
             linewidth=2, label='Improved Model', marker='s')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    # Add final test accuracy as text
    flawed_test_acc = results['flawed']['test_accuracy'] * 100
    improved_test_acc = results['improved']['test_accuracy'] * 100
    
    textstr = f'Final Test Accuracy:\n'
    textstr += f'Flawed: {flawed_test_acc:.2f}%\n'
    textstr += f'Improved: {improved_test_acc:.2f}%\n'
    textstr += f'Gain: +{improved_test_acc - flawed_test_acc:.2f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.05, 0.05, textstr, transform=ax4.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'results', 'training_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training comparison saved to: {output_path}")
    
    plt.show()


def plot_final_comparison_bar(results):
    """Create bar chart comparing final metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Final Performance Metrics: Flawed vs Improved', 
                 fontsize=16, fontweight='bold')
    
    flawed_acc = results['flawed']['test_accuracy'] * 100
    improved_acc = results['improved']['test_accuracy'] * 100
    flawed_loss = results['flawed']['test_loss']
    improved_loss = results['improved']['test_loss']
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    bars1 = ax1.bar(['Flawed Model', 'Improved Model'], 
                    [flawed_acc, improved_acc],
                    color=['#FF6B6B', '#51CF66'],
                    edgecolor='black',
                    linewidth=2)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Loss comparison
    ax2 = axes[1]
    bars2 = ax2.bar(['Flawed Model', 'Improved Model'], 
                    [flawed_loss, improved_loss],
                    color=['#FF6B6B', '#51CF66'],
                    edgecolor='black',
                    linewidth=2)
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'results', 'final_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Final metrics comparison saved to: {output_path}")
    
    plt.show()


def plot_convergence_speed(results):
    """Plot showing convergence speed difference"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    flawed = results['flawed']['history']
    improved = results['improved']['history']
    epochs = range(1, len(flawed['val_accuracy']) + 1)
    
    ax.plot(epochs, [acc*100 for acc in flawed['val_accuracy']], 'r-', 
            linewidth=3, label='Flawed Model (Slow Convergence)', marker='o', markersize=8)
    ax.plot(epochs, [acc*100 for acc in improved['val_accuracy']], 'g-', 
            linewidth=3, label='Improved Model (Fast Convergence)', marker='s', markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Convergence Speed Comparison\nDemonstrating Impact of Glorot Initialization & ReLU', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add annotations
    # Find when improved model reaches 95% accuracy
    improved_accs = [acc*100 for acc in improved['val_accuracy']]
    try:
        epoch_95 = next(i for i, acc in enumerate(improved_accs) if acc >= 95.0) + 1
        ax.axhline(y=95, color='blue', linestyle='--', alpha=0.5)
        ax.text(len(epochs)*0.7, 96, f'Improved reaches 95% at epoch {epoch_95}', 
                fontsize=11, color='green', fontweight='bold')
    except StopIteration:
        pass
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'results', 'convergence_speed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence speed plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main visualization function"""
    print("\n" + "="*70)
    print("VISUALIZING TRAINING RESULTS")
    print("="*70 + "\n")
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    print("📊 Generating plots...\n")
    
    # Generate all plots
    plot_training_comparison(results)
    plot_final_comparison_bar(results)
    plot_convergence_speed(results)
    
    print("\n" + "="*70)
    print("✅ All visualizations complete!")
    print("="*70)
    print("\n📁 All plots saved in the 'results' folder")
    print("   - training_comparison.png")
    print("   - final_metrics.png")
    print("   - convergence_speed.png\n")


if __name__ == "__main__":
    main()
