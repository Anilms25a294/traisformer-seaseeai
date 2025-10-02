"""
Compare LSTM vs Transformer performance on real AIS data
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from models.transformer_model import AISTransformer

def compare_models():
    print("=== Model Comparison: LSTM vs Transformer ===")
    
    # Load and prepare data
    df = pd.read_csv("AIS_2024_12_311.csv", nrows=3000)
    data = df[['LAT', 'LON', 'SOG', 'COG']].values
    
    # Create sequences
    sequences = []
    targets = []
    for i in range(len(data) - 8):
        sequences.append(data[i:i+5])
        targets.append(data[i+5:i+8])
    
    X = torch.FloatTensor(np.array(sequences))
    y = torch.FloatTensor(np.array(targets))
    
    print(f"Training data: {X.shape[0]} sequences")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Initialize models
    lstm_model = SimpleLSTMModel(input_size=4, hidden_size=32, prediction_length=3)
    transformer_model = AISTransformer(input_size=4, d_model=64, prediction_length=3)
    
    models = {
        'LSTM': lstm_model,
        'Transformer': transformer_model
    }
    
    # Train and compare
    results = {}
    criterion = nn.MSELoss()
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        for epoch in range(15):  # Quick training
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Validation
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        results[name] = {
            'train_loss': losses[-1],
            'val_loss': val_loss,
            'losses': losses
        }
        print(f"{name} - Final Train Loss: {losses[-1]:.6f}, Val Loss: {val_loss:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['losses'], label=name, linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    val_losses = [results[name]['val_loss'] for name in names]
    plt.bar(names, val_losses, alpha=0.7)
    plt.title('Validation Loss Comparison')
    plt.ylabel('MSE Loss')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Model comparison complete - check model_comparison.png")
    
    return results

if __name__ == "__main__":
    compare_models()
