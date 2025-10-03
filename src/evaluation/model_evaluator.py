"""
Comprehensive model evaluation for SeaSeeAI
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.traisformer import TrAISformer
from models.simple_lstm_model import SimpleLSTMModel
from data_processing.sample_data_generator import generate_realistic_ais_data
from data_processing.preprocessor import AISPreprocessor

class ModelEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def load_model(self, model_path, model_type='transformer'):
        """Load trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if model_type == 'transformer':
                model = TrAISformer(**checkpoint['config'])
            elif model_type == 'lstm':
                # For LSTM, we need to handle the output size properly
                config = checkpoint['config']
                model = SimpleLSTMModel(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    output_size=config['output_size'],
                    num_layers=config['num_layers'],
                    prediction_length=config['prediction_length']
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model, checkpoint['config']
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None, None
    
    def evaluate_model(self, model, config, test_samples=100):
        """Comprehensive model evaluation"""
        print(f"Evaluating model on {test_samples} test samples...")
        
        # Generate test data
        df = generate_realistic_ais_data(num_vessels=10, hours=48)
        preprocessor = AISPreprocessor()
        clean_df = preprocessor.clean_data(df)
        
        # Determine sequence length based on model type
        sequence_length = 20 if 'transformer' in str(type(model)).lower() else 10
        
        sequences, targets = preprocessor.create_sequences(
            clean_df, 
            sequence_length=sequence_length,
            prediction_length=config['prediction_length']
        )
        
        # Use last samples for testing
        test_sequences = sequences[-test_samples:]
        test_targets = targets[-test_samples:]
        
        predictions = []
        with torch.no_grad():
            for seq in test_sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = model(seq_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions).squeeze()
        
        # Calculate metrics
        mse = mean_squared_error(test_targets.reshape(-1, 4), predictions.reshape(-1, 4))
        mae = mean_absolute_error(test_targets.reshape(-1, 4), predictions.reshape(-1, 4))
        
        # Position-specific metrics (latitude and longitude)
        position_mse = mean_squared_error(
            test_targets[:, :, :2].reshape(-1, 2), 
            predictions[:, :, :2].reshape(-1, 2)
        )
        
        # Calculate distance error in kilometers
        true_positions = test_targets[:, :, :2].reshape(-1, 2)
        pred_positions = predictions[:, :, :2].reshape(-1, 2)
        
        distances = []
        for (true_lat, true_lon), (pred_lat, pred_lon) in zip(true_positions, pred_positions):
            distance = self.haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
            distances.append(distance)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'position_mse': position_mse,
            'mean_distance_error_km': np.mean(distances),
            'median_distance_error_km': np.median(distances),
            'max_distance_error_km': np.max(distances),
            'min_distance_error_km': np.min(distances),
            'std_distance_error_km': np.std(distances)
        }
        
        return metrics, test_targets, predictions, distances
    
    def compare_models(self, model_configs):
        """Compare multiple models"""
        comparison_results = {}
        
        for model_name, config in model_configs.items():
            print(f"\nEvaluating {model_name}...")
            
            model, model_config = self.load_model(config['path'], config['type'])
            if model is not None:
                metrics, _, _, distances = self.evaluate_model(model, model_config)
                comparison_results[model_name] = metrics
                comparison_results[model_name]['distances'] = distances
                
                print(f"  MSE: {metrics['mse']:.6f}")
                print(f"  Mean Distance Error: {metrics['mean_distance_error_km']:.2f} km")
                print(f"  Std Distance Error: {metrics['std_distance_error_km']:.2f} km")
            else:
                print(f"  Could not load model: {config['path']}")
        
        return comparison_results
    
    def plot_comparison(self, comparison_results, save_path='model_comparison.png'):
        """Create comprehensive comparison visualization"""
        if not comparison_results:
            print("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SeaSeeAI Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(comparison_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Plot 1: Mean Distance Error
        distance_errors = [results['mean_distance_error_km'] for results in comparison_results.values()]
        bars = axes[0, 0].bar(models, distance_errors, color=colors[:len(models)], alpha=0.8)
        axes[0, 0].set_title('Mean Distance Error (km)')
        axes[0, 0].set_ylabel('Kilometers')
        axes[0, 0].tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, value in zip(bars, distance_errors):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 2: MSE Comparison
        mse_values = [results['mse'] for results in comparison_results.values()]
        bars = axes[0, 1].bar(models, mse_values, color=colors[:len(models)], alpha=0.8)
        axes[0, 1].set_title('Mean Squared Error')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, mse_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 3: Distance Error Distribution
        for i, (model_name, results) in enumerate(comparison_results.items()):
            axes[1, 0].hist(results['distances'], alpha=0.7, label=model_name, 
                           bins=20, color=colors[i])
        axes[1, 0].set_title('Distance Error Distribution')
        axes[1, 0].set_xlabel('Distance Error (km)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error Statistics Boxplot
        distance_data = [results['distances'] for results in comparison_results.values()]
        box_plot = axes[1, 1].boxplot(distance_data, labels=models, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_title('Distance Error Statistics')
        axes[1, 1].set_ylabel('Distance Error (km)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved as '{save_path}'")
        
        return fig
    
    def generate_report(self, comparison_results):
        """Generate comprehensive evaluation report"""
        if not comparison_results:
            print("No results to generate report")
            return
            
        report = []
        report.append("=" * 60)
        report.append("SeaSeeAI Model Evaluation Report")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Find best model
        best_model = min(comparison_results.items(), 
                        key=lambda x: x[1]['mean_distance_error_km'])
        
        report.append("*** PERFORMANCE SUMMARY ***")
        report.append("-" * 40)
        report.append(f"Best Model: {best_model[0]}")
        report.append(f"Mean Distance Error: {best_model[1]['mean_distance_error_km']:.2f} km")
        report.append("")
        
        report.append("*** DETAILED METRICS ***")
        report.append("-" * 40)
        
        for model_name, metrics in comparison_results.items():
            report.append(f"\n{model_name.upper():<25}")
            report.append(f"  {'MSE:':<25} {metrics['mse']:.6f}")
            report.append(f"  {'MAE:':<25} {metrics['mae']:.6f}")
            report.append(f"  {'Position MSE:':<25} {metrics['position_mse']:.6f}")
            report.append(f"  {'Mean Distance Error:':<25} {metrics['mean_distance_error_km']:.2f} km")
            report.append(f"  {'Median Distance Error:':<25} {metrics['median_distance_error_km']:.2f} km")
            report.append(f"  {'Std Distance Error:':<25} {metrics['std_distance_error_km']:.2f} km")
            report.append(f"  {'Max Distance Error:':<25} {metrics['max_distance_error_km']:.2f} km")
            report.append(f"  {'Min Distance Error:':<25} {metrics['min_distance_error_km']:.2f} km")
        
        # Save report with UTF-8 encoding
        report_text = "\n".join(report)
        with open('model_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nFull report saved as 'model_evaluation_report.txt'")
        
        return report_text

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    # Define model configurations
    model_configs = {
        'LSTM Baseline': {
            'path': 'models/baseline_lstm.pth',
            'type': 'lstm'
        },
        'TrAISformer': {
            'path': 'models/best_traisformer.pth', 
            'type': 'transformer'
        }
    }
    
    print("Starting model evaluation...")
    
    # Compare models
    comparison_results = evaluator.compare_models(model_configs)
    
    if comparison_results:
        # Generate plots and report
        evaluator.plot_comparison(comparison_results)
        evaluator.generate_report(comparison_results)
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()
