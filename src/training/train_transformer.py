"""
Train Transformer model on AIS data with validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.transformer_model import TrAISformer
from data_processing.advanced_preprocessor import AdvancedAISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

def train_transformer():
    print("=== SeaSeeAI Transformer Training ===")
    
    # 1. Generate sample data
    print("1. Generating sample AIS data...")
    df = generate_realistic_ais_data(num_vessels=10, hours=72)
    
    # 2. Preprocess data with advanced preprocessor
    print("2. Preprocessing data...")
    preprocessor = AdvancedAISPreprocessor()
    processed_df = preprocessor.preprocess_data(df, vessel_id_col='mmsi', timestamp_col='timestamp')
    
    # 3. Create sequences
    sequences, targets = preprocessor.create_sequences(processed_df, sequence_length=10, prediction_length=5)
    
    # 4. Split data
    dataset = TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(targets))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 5. Create model
    print("3. Initializing Transformer model...")
    model = TrAISformer(input_size=4, d_model=64, nhead=8, num_layers=3, prediction_length=5)
    
    # 6. Train model
    print("4. Training model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_losses = []
    val_losses = []
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 7. Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Plot a sample prediction
    model.eval()
    with torch.no_grad():
        sample_X, sample_y = test_dataset[0]
        sample_X = sample_X.unsqueeze(0)
        prediction = model(sample_X)
        
        # Plot latitude and longitude
        plt.plot(sample_X[0, :, 0].numpy(), sample_X[0, :, 1].numpy(), 'bo-', label='Input Sequence')
        plt.plot(sample_y[:, 0].numpy(), sample_y[:, 1].numpy(), 'go-', label='Actual Future')
        plt.plot(prediction[0, :, 0].numpy(), prediction[0, :, 1].numpy(), 'ro-', label='Predicted Future')
        plt.title('Trajectory Prediction')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_training.png', dpi=150, bbox_inches='tight')
    print("Training plot saved as 'transformer_training.png'")
    
    # 8. Save model
    torch.save(model.state_dict(), 'models/transformer_ais.pth')
    print("Model saved as 'models/transformer_ais.pth'")
    
    print("✅ Transformer training completed!")
    return model, train_losses, val_losses

if __name__ == "__main__":
    train_transformer()
