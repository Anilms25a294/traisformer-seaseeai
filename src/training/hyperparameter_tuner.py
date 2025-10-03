"""
Hyperparameter tuning for SeaSeeAI models
"""

import torch
import torch.nn as nn
import numpy as np
import itertools
from pathlib import Path
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.traisformer import TrAISformer
from data_processing.preprocessor import AISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

class HyperparameterTuner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def get_search_space(self):
        """Define hyperparameter search space"""
        return {
            'd_model': [64, 128, 256],
            'nhead': [4, 8, 16],
            'num_layers': [2, 4, 6],
            'dim_feedforward': [256, 512, 1024],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64]
        }
    
    def generate_configs(self, search_space, max_configs=20):
        """Generate hyperparameter configurations"""
        keys = search_space.keys()
        values = search_space.values()
        
        all_combinations = list(itertools.product(*values))
        np.random.shuffle(all_combinations)
        
        configs = []
        for combo in all_combinations[:max_configs]:
            config = dict(zip(keys, combo))
            configs.append(config)
        
        return configs
    
    def train_single_config(self, config, epochs=50):
        """Train model with single configuration"""
        print(f"\nTraining config: {config}")
        
        try:
            # Prepare data
            df = generate_realistic_ais_data(num_vessels=8, hours=96)
            preprocessor = AISPreprocessor()
            clean_df = preprocessor.clean_data(df)
            sequences, targets = preprocessor.create_sequences(
                clean_df, sequence_length=20, prediction_length=5
            )
            
            # Split data
            split_idx = int(0.8 * len(sequences))
            X_train, y_train = sequences[:split_idx], targets[:split_idx]
            X_val, y_val = sequences[split_idx:], targets[split_idx:]
            
            # Create model
            model_config = {
                'input_size': 4,
                'd_model': config['d_model'],
                'nhead': config['nhead'],
                'num_layers': config['num_layers'],
                'dim_feedforward': config['dim_feedforward'],
                'dropout': config['dropout'],
                'prediction_length': 5
            }
            
            model = TrAISformer(**model_config)
            model.to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
            
            # Training loop
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for i in range(0, len(X_train), config['batch_size']):
                    batch_x = X_train[i:i + config['batch_size']]
                    batch_y = y_train[i:i + config['batch_size']]
                    
                    inputs = torch.FloatTensor(batch_x).to(self.device)
                    labels = torch.FloatTensor(batch_y).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= (len(X_train) // config['batch_size'] + 1)
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for i in range(0, len(X_val), config['batch_size']):
                        batch_x = X_val[i:i + config['batch_size']]
                        batch_y = y_val[i:i + config['batch_size']]
                        
                        inputs = torch.FloatTensor(batch_x).to(self.device)
                        labels = torch.FloatTensor(batch_y).to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                
                val_loss /= (len(X_val) // config['batch_size'] + 1)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            result = {
                'config': config,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"Best validation loss: {best_val_loss:.6f}")
            return result
            
        except Exception as e:
            print(f"Error training config {config}: {e}")
            return None
    
    def run_search(self, num_configs=10, epochs=30):
        """Run hyperparameter search"""
        print("Starting hyperparameter search...")
        search_space = self.get_search_space()
        configs = self.generate_configs(search_space, num_configs)
        
        for i, config in enumerate(configs):
            print(f"\n=== Config {i+1}/{len(configs)} ===")
            result = self.train_single_config(config, epochs)
            if result is not None:
                self.results.append(result)
        
        # Sort results by validation loss
        self.results.sort(key=lambda x: x['best_val_loss'])
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save hyperparameter search results"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            serializable_result['train_losses'] = [float(x) for x in result['train_losses']]
            serializable_result['val_losses'] = [float(x) for x in result['val_losses']]
            serializable_results.append(serializable_result)
        
        with open('hyperparameter_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save best configuration
        if self.results:
            best_config = self.results[0]['config']
            with open('best_hyperparameters.json', 'w') as f:
                json.dump(best_config, f, indent=2)
            
            print(f"\n🎯 Best configuration saved:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
            print(f"Best validation loss: {self.results[0]['best_val_loss']:.6f}")
    
    def plot_results(self):
        """Plot hyperparameter search results"""
        if not self.results:
            print("No results to plot")
            return
        
        import matplotlib.pyplot as plt
        
        # Plot top 5 configurations
        top_results = self.results[:5]
        
        plt.figure(figsize=(12, 8))
        
        for i, result in enumerate(top_results):
            plt.plot(result['train_losses'], label=f"Config {i+1} (Train)", alpha=0.7)
            plt.plot(result['val_losses'], label=f"Config {i+1} (Val)", linestyle='--', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hyperparameter Search - Top 5 Configurations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('hyperparameter_search.png', dpi=300, bbox_inches='tight')
        print("Hyperparameter search plot saved as 'hyperparameter_search.png'")

def main():
    """Run hyperparameter tuning"""
    tuner = HyperparameterTuner()
    results = tuner.run_search(num_configs=8, epochs=25)  # Reduced for demo
    
    if results:
        tuner.plot_results()
        print(f"\nHyperparameter search completed. Evaluated {len(results)} configurations.")
    else:
        print("Hyperparameter search failed.")

if __name__ == "__main__":
    main()
