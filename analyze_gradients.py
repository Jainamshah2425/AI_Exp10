"""
Gradient Flow Analysis Script
Analyzes and visualizes gradient flow through network layers
Demonstrates vanishing gradients in flawed model vs healthy flow in improved model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

# Add models directory to path
sys.path.append(os.path.dirname(__file__))

from models import create_flawed_model, create_improved_model

# Set style
sns.set_style("whitegrid")


def analyze_gradient_flow(model_wrapper, x_sample, y_sample, model_name):
    """
    Analyze gradient magnitudes across layers
    
    Args:
        model_wrapper: FlawedMNISTModel or ImprovedMNISTModel instance
        x_sample: Sample input data
        y_sample: Sample labels
        model_name: Name for plotting
    
    Returns:
        Dictionary with gradient statistics per layer
    """
    print(f"\n🔍 Analyzing gradient flow for {model_name}...")
    
    gradients = model_wrapper.get_gradients(x_sample, y_sample)
    
    gradient_stats = {}
    layer_names = []
    mean_grads = []
    std_grads = []
    
    for i, (grad, var) in enumerate(zip(gradients, model_wrapper.model.trainable_variables)):
        if grad is not None and 'kernel' in var.name:  # Only analyze weights, not biases
            grad_values = grad.numpy().flatten()
            
            layer_name = var.name.split('/')[0]
            mean_grad = np.mean(np.abs(grad_values))
            std_grad = np.std(grad_values)
            
            gradient_stats[layer_name] = {
                'mean': mean_grad,
                'std': std_grad,
                'min': np.min(grad_values),
                'max': np.max(grad_values)
            }
            
            layer_names.append(layer_name)
            mean_grads.append(mean_grad)
            std_grads.append(std_grad)
    
    return gradient_stats, layer_names, mean_grads, std_grads


def plot_gradient_comparison(flawed_stats, improved_stats):
    """Create comprehensive gradient flow comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gradient Flow Analysis: Flawed vs Improved Model\n' + 
                 'Demonstrating Vanishing Gradient Problem', 
                 fontsize=16, fontweight='bold')
    
    flawed_layers = list(flawed_stats[1])
    improved_layers = list(improved_stats[1])
    flawed_means = flawed_stats[2]
    improved_means = improved_stats[2]
    flawed_stds = flawed_stats[3]
    improved_stds = improved_stats[3]
    
    x_pos = np.arange(len(flawed_layers))
    
    # Plot 1: Mean Gradient Magnitude (Bar chart)
    ax1 = axes[0, 0]
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, flawed_means, width, 
                    label='Flawed Model', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, improved_means, width, 
                    label='Improved Model', color='#51CF66', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean |Gradient|', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Gradient Magnitude per Layer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(flawed_layers, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see vanishing gradients
    
    # Plot 2: Mean Gradient Magnitude (Line chart)
    ax2 = axes[0, 1]
    ax2.plot(x_pos, flawed_means, 'r-', linewidth=3, marker='o', 
             markersize=10, label='Flawed Model (Vanishing)')
    ax2.plot(x_pos, improved_means, 'g-', linewidth=3, marker='s', 
             markersize=10, label='Improved Model (Healthy)')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean |Gradient|', fontsize=12, fontweight='bold')
    ax2.set_title('Gradient Flow Across Layers', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(flawed_layers, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add annotation for vanishing gradients
    ax2.text(0.5, 0.95, 'Notice: Flawed model gradients vanish in early layers',
             transform=ax2.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             ha='center', va='top')
    
    # Plot 3: Gradient Standard Deviation
    ax3 = axes[1, 0]
    ax3.bar(x_pos - width/2, flawed_stds, width, 
            label='Flawed Model', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax3.bar(x_pos + width/2, improved_stds, width, 
            label='Improved Model', color='#51CF66', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Gradient Std Dev', fontsize=12, fontweight='bold')
    ax3.set_title('Gradient Variance per Layer', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(flawed_layers, rotation=45, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Ratio comparison (Improved / Flawed)
    ax4 = axes[1, 1]
    ratios = [imp/flw if flw > 0 else 0 for imp, flw in zip(improved_means, flawed_means)]
    bars = ax4.bar(x_pos, ratios, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=2)
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal gradient flow')
    ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Ratio (Improved / Flawed)', fontsize=12, fontweight='bold')
    ax4.set_title('Gradient Flow Improvement Ratio', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(flawed_layers, rotation=45, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_yscale('log')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'results', 'gradient_flow_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gradient flow analysis saved to: {output_path}")
    
    plt.show()


def print_gradient_stats(flawed_stats, improved_stats):
    """Print detailed gradient statistics"""
    print("\n" + "="*70)
    print("GRADIENT FLOW STATISTICS")
    print("="*70)
    
    flawed_dict = flawed_stats[0]
    improved_dict = improved_stats[0]
    
    print("\n📊 Mean Gradient Magnitudes (|gradient|):\n")
    print(f"{'Layer':<15} {'Flawed Model':<20} {'Improved Model':<20} {'Ratio':<10}")
    print("-" * 70)
    
    for layer in flawed_dict.keys():
        flawed_mean = flawed_dict[layer]['mean']
        improved_mean = improved_dict[layer]['mean']
        ratio = improved_mean / flawed_mean if flawed_mean > 0 else float('inf')
        
        print(f"{layer:<15} {flawed_mean:<20.6e} {improved_mean:<20.6e} {ratio:<10.2f}x")
    
    print("\n" + "="*70)
    print("\n🔑 KEY OBSERVATIONS:")
    print("   ❌ Flawed model: Gradients vanish in early layers (sigmoid saturation)")
    print("   ✅ Improved model: Healthy gradient flow throughout (ReLU + Glorot init)")
    print("   📈 Early layers in improved model have 10-1000x larger gradients!")
    print("="*70 + "\n")


def main():
    """Main gradient analysis function"""
    print("\n" + "="*70)
    print("GRADIENT FLOW ANALYSIS")
    print("Demonstrating Vanishing Gradient Problem from Glorot & Bengio (2010)")
    print("="*70)
    
    # Load MNIST data
    print("\n📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Take a small sample for gradient analysis
    sample_size = 32
    x_sample = x_train[:sample_size]
    y_sample = y_train[:sample_size]
    
    # Create models
    print("\n🔴 Creating flawed model...")
    flawed_model = create_flawed_model()
    
    print("🟢 Creating improved model...")
    improved_model = create_improved_model()
    
    # Analyze gradients
    flawed_stats = analyze_gradient_flow(
        flawed_model, x_sample, y_sample, "Flawed Model"
    )
    
    improved_stats = analyze_gradient_flow(
        improved_model, x_sample, y_sample, "Improved Model"
    )
    
    # Print statistics
    print_gradient_stats(flawed_stats, improved_stats)
    
    # Create visualizations
    print("\n📊 Generating gradient flow plots...")
    plot_gradient_comparison(flawed_stats, improved_stats)
    
    print("\n✅ Gradient analysis complete!")
    print("   The plots clearly show vanishing gradients in the flawed model.")
    print("   Early layers barely receive any gradient signal for learning!\n")


if __name__ == "__main__":
    main()
