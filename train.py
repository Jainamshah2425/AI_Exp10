"""
Main Training Script
Trains both flawed and improved models and saves results for comparison
"""

import os
import sys
import pickle
import numpy as np
from tensorflow import keras

# Add models directory to path
sys.path.append(os.path.dirname(__file__))

from models import create_flawed_model, create_improved_model


def load_mnist_data():
    """Load and return MNIST dataset"""
    print("\n📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"   Training samples: {len(x_train)}")
    print(f"   Test samples: {len(x_test)}")
    print(f"   Image shape: {x_train[0].shape}")
    
    return (x_train, y_train), (x_test, y_test)


def save_results(flawed_history, improved_history, flawed_eval, improved_eval):
    """Save training results to disk"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'flawed': {
            'history': flawed_history.history,
            'test_loss': flawed_eval[0],
            'test_accuracy': flawed_eval[1]
        },
        'improved': {
            'history': improved_history.history,
            'test_loss': improved_eval[0],
            'test_accuracy': improved_eval[1]
        }
    }
    
    results_path = os.path.join(results_dir, 'training_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {results_path}")


def print_comparison(flawed_eval, improved_eval):
    """Print side-by-side comparison of results"""
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    
    print("\n📊 Test Accuracy:")
    print(f"   Flawed Model:   {flawed_eval[1]*100:.2f}%")
    print(f"   Improved Model: {improved_eval[1]*100:.2f}%")
    print(f"   Improvement:    +{(improved_eval[1] - flawed_eval[1])*100:.2f}%")
    
    print("\n📉 Test Loss:")
    print(f"   Flawed Model:   {flawed_eval[0]:.4f}")
    print(f"   Improved Model: {improved_eval[0]:.4f}")
    print(f"   Reduction:      {(flawed_eval[0] - improved_eval[0]):.4f}")
    
    print("\n" + "="*70)
    
    # Print key takeaways
    print("\n🔑 KEY TAKEAWAYS:")
    print("   ✅ Glorot/Xavier initialization prevents vanishing gradients")
    print("   ✅ ReLU activation enables better gradient flow")
    print("   ✅ Input normalization improves convergence speed")
    print("   ✅ Proper learning rate + Adam optimizer stabilizes training")
    print("="*70 + "\n")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("MNIST Deep Learning Training - Flaw Demonstration & Fix")
    print("Based on: Glorot & Bengio (2010)")
    print("="*70)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Training parameters
    EPOCHS = 15
    BATCH_SIZE = 128
    
    # ============================================================
    # PHASE 1: Train Flawed Model
    # ============================================================
    print("\n" + "🔴" * 35)
    print("PHASE 1: FLAWED MODEL")
    print("🔴" * 35)
    
    flawed_model = create_flawed_model()
    flawed_model.model.summary()
    
    flawed_history = flawed_model.train(
        x_train, y_train, x_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    flawed_eval = flawed_model.evaluate(x_test, y_test)
    print(f"\n🔴 Flawed Model - Test Accuracy: {flawed_eval[1]*100:.2f}%")
    
    # ============================================================
    # PHASE 2: Train Improved Model
    # ============================================================
    print("\n\n" + "🟢" * 35)
    print("PHASE 2: IMPROVED MODEL")
    print("🟢" * 35)
    
    improved_model = create_improved_model()
    improved_model.model.summary()
    
    improved_history = improved_model.train(
        x_train, y_train, x_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    improved_eval = improved_model.evaluate(x_test, y_test)
    print(f"\n🟢 Improved Model - Test Accuracy: {improved_eval[1]*100:.2f}%")
    
    # ============================================================
    # PHASE 3: Comparison & Save Results
    # ============================================================
    print_comparison(flawed_eval, improved_eval)
    save_results(flawed_history, improved_history, flawed_eval, improved_eval)
    
    # Save models
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    flawed_model.model.save(os.path.join(results_dir, 'flawed_model.h5'))
    improved_model.model.save(os.path.join(results_dir, 'improved_model.h5'))
    print(f"💾 Models saved to: {results_dir}")
    
    print("\n✅ Training complete! Run 'python visualize.py' to see visual comparisons.")
    print("✅ Run 'python analyze_gradients.py' to analyze gradient flow.\n")


if __name__ == "__main__":
    main()
