"""
Flawed MNIST Model - Demonstrating Training Difficulties
Based on issues discussed in Glorot & Bengio (2010)

This model intentionally implements BAD practices:
1. Poor weight initialization (random uniform, not Xavier/Glorot)
2. Sigmoid activation (causes vanishing gradients)
3. No input normalization (raw pixel values 0-255)
4. Poor learning rate
5. Deep architecture makes problems worse
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class FlawedMNISTModel:
    """
    A deliberately flawed deep neural network for MNIST classification.
    Demonstrates the issues discussed in Glorot & Bengio's paper.
    """
    
    def __init__(self, input_shape=(28, 28)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build a deep neural network with INTENTIONAL FLAWS:
        - Poor initialization
        - Sigmoid activations (saturate easily)
        - Deep architecture (5 hidden layers)
        """
        model = keras.Sequential([
            layers.Flatten(input_shape=self.input_shape),
            
            # Layer 1 - Poor initialization + Sigmoid
            layers.Dense(
                512,
                activation='sigmoid',
                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                name='hidden_1'
            ),
            
            # Layer 2
            layers.Dense(
                256,
                activation='sigmoid',
                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                name='hidden_2'
            ),
            
            # Layer 3
            layers.Dense(
                128,
                activation='sigmoid',
                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                name='hidden_3'
            ),
            
            # Layer 4
            layers.Dense(
                64,
                activation='sigmoid',
                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                name='hidden_4'
            ),
            
            # Layer 5
            layers.Dense(
                32,
                activation='sigmoid',
                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                name='hidden_5'
            ),
            
            # Output layer
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self):
        """
        Compile with poor settings:
        - High learning rate (0.1) - too high for deep networks
        - Basic SGD without momentum
        """
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1),  # Too high!
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, x_train, y_train, x_test, y_test):
        """
        FLAW: No normalization - using raw pixel values (0-255)
        This causes poor convergence
        """
        # Intentionally NOT normalizing!
        return x_train, y_train, x_test, y_test
    
    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
        """Train the flawed model"""
        x_train_prep, y_train_prep, x_test_prep, y_test_prep = self.prepare_data(
            x_train, y_train, x_test, y_test
        )
        
        print("\n" + "="*60)
        print("Training FLAWED Model")
        print("="*60)
        print("INTENTIONAL FLAWS:")
        print("  ❌ Poor weight initialization (Random Uniform)")
        print("  ❌ Sigmoid activation (vanishing gradients)")
        print("  ❌ No input normalization (raw 0-255 values)")
        print("  ❌ High learning rate (0.1)")
        print("  ❌ Deep architecture (5 hidden layers)")
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
        This will show vanishing gradients in early layers.
        """
        x_sample_prep, _, _, _ = self.prepare_data(x_sample, None, None, None)
        
        with tf.GradientTape() as tape:
            predictions = self.model(x_sample_prep, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(y_sample, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients


def create_flawed_model():
    """Factory function to create and compile a flawed model"""
    model = FlawedMNISTModel()
    model.build_model()
    model.compile_model()
    return model


if __name__ == "__main__":
    # Quick test
    print("Flawed Model Architecture:")
    model = create_flawed_model()
    model.model.summary()
