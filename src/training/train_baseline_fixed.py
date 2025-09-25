"""
Train the baseline LSTM model on AIS data - FIXED VERSION
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from data_processing.preprocessor import AISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

def train_model():
    print("=== SeaSeeAI Baseline Model Training ===")
    
    # 1. Generate sample data
    print("1. Generating sample AIS data...")
    df = generate_realistic_ais_data(num_vessels=5, hours=48)
    
    # 2. Preprocess data
    print("2. Preprocessing data...")
    preprocessor = AISPreprocessor()
    clean_df = preprocessor.clean_data(df)
    sequences, targets = preprocessor.create_sequences(clean_df, sequence_length=10, prediction_length=5)
    
    print("Input sequences shape:", sequences.shape)
    print("Target sequences shape:", targets.shape)
    
    # 3. Split data (simple split for demo)
    split_idx = int(0.8 * len(sequences))
    X_train, y_train = sequences[:split_idx], targets[:split_idx]
    X_test, y_test = sequences[split_idx:], targets[split_idx:]
    
    print("Training samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    
    # 4. Create model with correct output shape
    print("3. Initializing LSTM model...")
    model = SimpleLSTMModel(
        input_size=4, 
        hidden_size=64, 
        output_size=4, 
        num_layers=2,
        prediction_length=5
    )
    
    # Test the model shape before training
    test_input = torch.FloatTensor(X_train[:1])
    test_output = model(test_input)
    print("Model input shape:", test_input.shape)
    print("Model output shape:", test_output.shape)
    print("Target shape:", y_train[:1].shape)
    
    # Check if shapes match
    if test_output.shape[1:] == y_train[:1].shape[1:]:
        print("Shape check passed! Ready for training.")
    else:
        print("Shape mismatch! Fix the model.")
        return
    
    # 5. Train model
    print("4. Training model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Convert to tensors
        inputs = torch.FloatTensor(X_train)
        labels = torch.FloatTensor(y_train)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print('Epoch', epoch, '/', num_epochs, 'Loss:', loss.item())
    
    # 6. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('SeaSeeAI - LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Training plot saved as training_loss.png")
    
    # 7. Save model
    torch.save(model.state_dict(), 'models/baseline_lstm.pth')
    print("Model saved as models/baseline_lstm.pth")
    
    print("Training completed successfully!")
    return model, train_losses

if __name__ == "__main__":
    train_model()
