"""
Improved MNIST Model - Applying Fixes from Glorot & Bengio (2010)
This model implements BEST practices to fix the training difficulties

FIXES APPLIED:
1. Xavier/Glorot initialization (proper weight scaling)
2. ReLU activation (no saturation, better gradient flow)
3. Input normalization (scale to 0-1)
4. Proper learning rate with Adam optimizer
5. Same deep architecture to show fair comparison
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ImprovedMNISTModel:
    """
    An improved deep neural network for MNIST classification.
    Implements fixes discussed in Glorot & Bengio's paper.
    """
    
    def __init__(self, input_shape=(28, 28)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build a deep neural network with BEST PRACTICES:
        - Xavier/Glorot initialization
        - ReLU activations (no vanishing gradients)
        - Same architecture as flawed model for fair comparison
        """
        model = keras.Sequential([
            layers.Flatten(input_shape=self.input_shape),
            
            # Layer 1 - Glorot initialization + ReLU
            layers.Dense(
                512,
                activation='relu',
                kernel_initializer='glorot_uniform',  # Xavier/Glorot init!
                name='hidden_1'
            ),
            
            # Layer 2
            layers.Dense(
                256,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name='hidden_2'
            ),
            
            # Layer 3
            layers.Dense(
                128,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name='hidden_3'
            ),
            
            # Layer 4
            layers.Dense(
                64,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name='hidden_4'
            ),
            
            # Layer 5
            layers.Dense(
                32,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name='hidden_5'
            ),
            
            # Output layer
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self):
        """
        Compile with IMPROVED settings:
        - Proper learning rate (0.001)
        - Adam optimizer (adaptive learning rate)
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Much better!
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, x_train, y_train, x_test, y_test):
        """
        FIX: Normalize input to [0, 1] range
        This significantly improves convergence
        """
        if x_train is not None:
            x_train = x_train.astype('float32') / 255.0
        if x_test is not None:
            x_test = x_test.astype('float32') / 255.0
        
        return x_train, y_train, x_test, y_test
    
    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
        """Train the improved model"""
        x_train_prep, y_train_prep, x_test_prep, y_test_prep = self.prepare_data(
            x_train, y_train, x_test, y_test
        )
        
        print("\n" + "="*60)
        print("Training IMPROVED Model")
        print("="*60)
        print("FIXES APPLIED:")
        print("  ✅ Xavier/Glorot initialization")
        print("  ✅ ReLU activation (no vanishing gradients)")
        print("  ✅ Input normalization (scaled to 0-1)")
        print("  ✅ Proper learning rate (0.001)")
        print("  ✅ Adam optimizer (adaptive learning)")
        print("="*60 + "\n")
        
        self.history = self.model.fit(
            x_train_prep, y_train_prep,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test_prep, y_test_prep),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """Evaluate the model"""
        x_test_prep, _, _, _ = self.prepare_data(x_test, None, None, None)
        return self.model.evaluate(x_test_prep, y_test, verbose=0)
    
    def get_gradients(self, x_sample, y_sample):
        """
        Compute gradients for gradient flow analysis.
        This will show HEALTHY gradients throughout all layers.
        """
        x_sample_prep, _, _, _ = self.prepare_data(x_sample, None, None, None)
        
        with tf.GradientTape() as tape:
            predictions = self.model(x_sample_prep, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(y_sample, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients


def create_improved_model():
    """Factory function to create and compile an improved model"""
    model = ImprovedMNISTModel()
    model.build_model()
    model.compile_model()
    return model


if __name__ == "__main__":
    # Quick test
    print("Improved Model Architecture:")
    model = create_improved_model()
    model.model.summary()
