"""
Training script for TrAISformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.traisformer import TrAISformer
from data_processing.preprocessor import AISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

class TraisformerTrainer:
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model - only model parameters, no training hyperparameters
        self.model = TrAISformer(
            input_size=self.config['input_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            prediction_length=self.config['prediction_length']
        )

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = nn.MSELoss()
        
    @staticmethod
    def get_default_config():
        return {
            'input_size': 4,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 5,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'epochs': 50,  # Reduced for faster training
            'batch_size': 32
        }
    
    def prepare_data(self):
        """Prepare training and validation data"""
        print("Generating training data...")
        df = generate_realistic_ais_data(num_vessels=8, hours=72)
        
        preprocessor = AISPreprocessor()
        clean_df = preprocessor.clean_data(df)
        sequences, targets = preprocessor.create_sequences(
            clean_df, 
            sequence_length=20,
            prediction_length=self.config['prediction_length']
        )
        
        # Split data
        split_idx = int(0.8 * len(sequences))
        X_train, y_train = sequences[:split_idx], targets[:split_idx]
        X_val, y_val = sequences[split_idx:], targets[split_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        return train_loader, val_loader, X_train.shape[1], X_train.shape[2]
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self):
        """Main training loop"""
        print("=== SeaSeeAI TrAISformer Training ===")
        print(f"Config: {self.config}")
        
        train_loader, val_loader, seq_len, input_size = self.prepare_data()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"Starting training for {self.config['epochs']} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_traisformer.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}/{self.config["epochs"]:03d}:')
                print(f'  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_model('final_traisformer.pth')
        
        # Plot results
        self.plot_training(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def save_model(self, filename):
        """Save model to file - only model parameters, no training config"""
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        # Only save model architecture parameters, not training hyperparameters
        model_config = {
            'input_size': self.config['input_size'],
            'd_model': self.config['d_model'],
            'nhead': self.config['nhead'],
            'num_layers': self.config['num_layers'],
            'dim_feedforward': self.config['dim_feedforward'],
            'dropout': self.config['dropout'],
            'prediction_length': self.config['prediction_length']
        }
        
        model_path = model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': model_config  # Only model architecture config
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def plot_training(self, train_losses, val_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('TrAISformer Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(train_losses, label='Training Loss')
        plt.semilogy(val_losses, label='Validation Loss')
        plt.title('Training Progress (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (log)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('traisformer_training.png', dpi=300, bbox_inches='tight')
        print("Training plot saved as 'traisformer_training.png'")

def compare_models():
    """Compare LSTM vs Transformer performance"""
    print("=== Model Comparison: LSTM vs Transformer ===")
    
    # Test data
    test_input = torch.randn(1, 20, 4)
    
    # LSTM model
    from models.simple_lstm_model import SimpleLSTMModel
    lstm_model = SimpleLSTMModel(input_size=4, hidden_size=64, output_size=20, num_layers=2)
    lstm_output = lstm_model(test_input)
    
    # Transformer model
    transformer_model = TrAISformer(input_size=4, prediction_length=5)
    transformer_output = transformer_model(test_input)
    
    print(f"LSTM Model:")
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"  Output shape: {lstm_output.shape}")
    
    print(f"Transformer Model:")
    print(f"  Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    print(f"  Output shape: {transformer_output.shape}")

if __name__ == "__main__":
    # Compare models first
    compare_models()
    
    # Train transformer
    print("\n" + "="*50)
    trainer = TraisformerTrainer()
    trainer.train()
