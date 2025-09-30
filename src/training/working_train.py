"""
Working training script for SeaSeeAI
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from data_processing.preprocessor import AISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

# Simple working model
class WorkingModel(nn.Module):
    def __init__(self, seq_len=10, pred_len=5, features=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len * features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * features)
        )
        self.pred_len = pred_len
        self.features = features
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        out = self.fc(x)
        return out.reshape(batch_size, self.pred_len, self.features)

def train_working_model():
    print("=== SeaSeeAI Working Training ===")
    
    # Generate sample data
    print("1. Generating data...")
    df = generate_realistic_ais_data(num_vessels=5, hours=48)
    preprocessor = AISPreprocessor()
    clean_df = preprocessor.clean_data(df)
    sequences, targets = preprocessor.create_sequences(clean_df)
    
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets)
    
    print(f"Training data: {X.shape}")
    print(f"Target data: {y.shape}")
    
    # Create model
    model = WorkingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("2. Training model...")
    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SeaSeeAI - Working Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('working_training.png')
    print("3. Training plot saved as 'working_training.png'")
    
    # Save model
    torch.save(model.state_dict(), 'models/working_model.pth')
    print("4. Model saved as 'models/working_model.pth'")
    
    print("✅ Training completed successfully!")
    return model, losses

if __name__ == "__main__":
    train_working_model()
