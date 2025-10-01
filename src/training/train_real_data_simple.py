"""
Train LSTM model on real AIS data (simplified version)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from data_processing.preprocessor import AISPreprocessor

def train_on_real_data_simple():
    print("=== SeaSeeAI Real Data Training (LSTM Only) ===")
    
    # 1. Load processed real data
    print("1. Loading real AIS data...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        print(f"   Loaded {len(real_df):,} real AIS records")
        print(f"   {real_df['mmsi'].nunique()} unique vessels")
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return
    
    # 2. Use a subset for faster training (first 50,000 records)
    print("2. Using subset for faster training...")
    subset_df = real_df.head(50000)
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
    
    # 4. Train LSTM on real data
    print("4. Training LSTM on real data...")
    model = SimpleLSTMModel(
        input_size=4, 
        hidden_size=64, 
        output_size=4, 
        prediction_length=5
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    num_epochs = 50  # Reduced for faster training
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        inputs = torch.FloatTensor(sequences)
        labels = torch.FloatTensor(targets)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f'   Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # Save model trained on real data
    torch.save(model.state_dict(), 'models/lstm_real_data.pth')
    
    # 5. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SeaSeeAI - LSTM Training on Real AIS Data')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('real_data_training.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data training completed!")
    print(f"   Model saved: models/lstm_real_data.pth")
    print(f"   Training plot: real_data_training.png")
    print(f"   Training sequences: {len(sequences)}")
    print(f"   Final loss: {losses[-1]:.6f}")
    
    return model, losses

if __name__ == "__main__":
    train_on_real_data_simple()
