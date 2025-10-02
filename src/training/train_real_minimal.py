"""
Minimal training on real AIS data (safe for low compute)
"""

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel

def load_real_data_sample(file_path, sample_size=2000):
    """Load small sample of real data with correct column names"""
    print(f"Loading {sample_size} samples from real data...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Map real column names to expected names
    column_mapping = {
        'LAT': 'latitude',
        'LON': 'longitude', 
        'SOG': 'sog',
        'COG': 'cog'
    }
    
    # Check for required columns in the real data
    available_cols = []
    for real_col, expected_col in column_mapping.items():
        if real_col in df.columns:
            df[expected_col] = df[real_col]
            available_cols.append(expected_col)
    
    print(f"Available columns after mapping: {available_cols}")
    
    if len(available_cols) < 4:
        print(f"Missing some columns. Only found: {available_cols}")
        return None
    
    return df[available_cols]

def create_sequences_simple(data, seq_length=5, pred_length=3):
    """Create sequences without complex preprocessing"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - pred_length):
        seq = data.iloc[i:i+seq_length].values
        target = data.iloc[i+seq_length:i+seq_length+pred_length].values
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_minimal_real_data():
    print("=== SeaSeeAI Real Data Training (Minimal) ===")
    
    # 1. Load small sample of real data
    real_data = load_real_data_sample("AIS_2024_12_311.csv", sample_size=2000)
    if real_data is None:
        print("❌ Could not load real data with required columns")
        return
    
    print(f"Loaded real data shape: {real_data.shape}")
    print(f"Data sample:\n{real_data.head(3)}")
    
    # 2. Create simple sequences
    sequences, targets = create_sequences_simple(real_data, seq_length=5, pred_length=3)
    
    if len(sequences) == 0:
        print("❌ No sequences created")
        return
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}, Target shape: {targets.shape}")
    
    # 3. Train model with correct prediction length
    model = SimpleLSTMModel(
        input_size=4, 
        hidden_size=16,  # Very small for quick training
        output_size=4, 
        num_layers=1,
        prediction_length=3  # Match our target length
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Quick training (few epochs)
    losses = []
    num_epochs = 10
    
    print("Starting training...")
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
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # 5. Quick plot
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Real Data Training Loss (Minimal)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('real_data_minimal_training.png', dpi=100, bbox_inches='tight')
    
    print("✅ Minimal real data training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        sample_input = torch.FloatTensor(sequences[:1])
        sample_output = model(sample_input)
        print(f"Sample prediction shape: {sample_output.shape}")
        print(f"Matches target shape: {sample_output.shape == targets[:1].shape}")
    
    return model, losses

if __name__ == "__main__":
    train_minimal_real_data()
