"""
Train models on real AIS data
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
from models.transformer import TrAISformer
from data_processing.preprocessor import AISPreprocessor

def train_on_real_data():
    print("=== SeaSeeAI Real Data Training ===")
    
    # 1. Load processed real data
    print("1. Loading real AIS data...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        print(f"   Loaded {len(real_df):,} real AIS records")
        print(f"   {real_df['mmsi'].nunique()} unique vessels")
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("   Please run real_data_processor.py first")
        return
    
    # 2. Preprocess real data
    print("2. Preprocessing real data...")
    preprocessor = AISPreprocessor()
    
    # Clean and create sequences
    clean_df = preprocessor.clean_data(real_df)
    sequences, targets = preprocessor.create_sequences(clean_df)
    
    print(f"   Created {len(sequences)} sequences from real data")
    
    if len(sequences) == 0:
        print("❌ No sequences created. Check data quality.")
        return
    
    # 3. Train LSTM on real data
    print("3. Training LSTM on real data...")
    lstm_model = SimpleLSTMModel(
        input_size=4, 
        hidden_size=64, 
        output_size=4, 
        prediction_length=5
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    lstm_losses = []
    for epoch in range(100):
        lstm_model.train()
        optimizer.zero_grad()
        
        inputs = torch.FloatTensor(sequences)
        labels = torch.FloatTensor(targets)
        
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        lstm_losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'   Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # Save LSTM trained on real data
    torch.save(lstm_model.state_dict(), 'models/lstm_real_data.pth')
    
    # 4. Train Transformer on real data
    print("4. Training Transformer on real data...")
    transformer_model = TrAISformer(
        input_size=4,
        d_model=128,
        nhead=8, 
        num_layers=4,
        prediction_length=5
    )
    
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
    
    transformer_losses = []
    for epoch in range(100):
        transformer_model.train()
        optimizer.zero_grad()
        
        inputs = torch.FloatTensor(sequences)
        labels = torch.FloatTensor(targets)
        
        outputs = transformer_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        transformer_losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'   Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # Save Transformer trained on real data
    torch.save(transformer_model.state_dict(), 'models/transformer_real_data.pth')
    
    # 5. Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_losses, label='LSTM Real Data', color='blue')
    plt.plot(transformer_losses, label='Transformer Real Data', color='red')
    plt.title('Training on Real AIS Data')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Compare with synthetic data training if available
    try:
        # This would require having synthetic training losses saved
        plt.text(0.3, 0.5, 'Real Data Training\nCompleted Successfully!', 
                fontsize=16, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    except:
        plt.text(0.5, 0.5, f'Real Data Results:\n{len(sequences)} sequences\n{real_df["mmsi"].nunique()} vessels', 
                fontsize=12, ha='center', va='center')
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('real_data_training.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data training completed!")
    print(f"   Models saved:")
    print(f"   - LSTM: models/lstm_real_data.pth")
    print(f"   - Transformer: models/transformer_real_data.pth")
    print(f"   - Plot: real_data_training.png")
    print(f"   - Training sequences: {len(sequences)}")
    
    return lstm_model, transformer_model, lstm_losses, transformer_losses

if __name__ == "__main__":
    train_on_real_data()
