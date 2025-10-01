"""
Train LSTM on real AIS data with memory safety
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from data_processing.preprocessor import AISPreprocessor

def train_on_real_data_memory_safe():
    print("=== SeaSeeAI Real Data Training (Memory Safe) ===")
    
    # 1. Load processed real data
    print("1. Loading real AIS data...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        print(f"   Loaded {len(real_df):,} real AIS records")
        print(f"   {real_df['mmsi'].nunique()} unique vessels")
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return
    
    # 2. Use a much smaller subset for initial testing
    print("2. Using small subset for memory safety...")
    subset_df = real_df.head(10000)  # Only 10,000 records
    print(f"   Using {len(subset_df):,} records for training")
    
    # 3. Preprocess real data
    print("3. Preprocessing real data...")
    preprocessor = AISPreprocessor()
    
    # Clean and create sequences
    clean_df = preprocessor.clean_data(subset_df)
    sequences, targets = preprocessor.create_sequences(clean_df)
    
    print(f"   Created {len(sequences)} sequences from real data")
    
    if len(sequences) == 0:
        print("❌ No sequences created. Check data quality.")
        return
    
    # 4. Use DataLoader for batch training
    print("4. Setting up batch training...")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(sequences)
    y_tensor = torch.FloatTensor(targets)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Small batches
    
    # 5. Train LSTM on real data with batches
    print("5. Training LSTM on real data (with batches)...")
    model = SimpleLSTMModel(
        input_size=4, 
        hidden_size=64, 
        output_size=4, 
        prediction_length=5
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    num_epochs = 30  # Reduced for faster training
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            model.train()
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        
        if epoch % 5 == 0:
            print(f'   Epoch {epoch}, Loss: {avg_epoch_loss:.6f}')
    
    # Save model trained on real data
    torch.save(model.state_dict(), 'models/lstm_real_data.pth')
    
    # 6. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SeaSeeAI - LSTM Training on Real AIS Data\n(Memory Safe Batch Training)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('real_data_training.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data training completed!")
    print(f"   Model saved: models/lstm_real_data.pth")
    print(f"   Training plot: real_data_training.png")
    print(f"   Training sequences: {len(sequences)}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Batch size: 32, Epochs: {num_epochs}")
    
    return model, losses

if __name__ == "__main__":
    train_on_real_data_memory_safe()
