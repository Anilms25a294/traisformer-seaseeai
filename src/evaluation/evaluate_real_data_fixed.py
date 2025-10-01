"""
Evaluate LSTM model on real AIS data (compatible with smaller model)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from data_processing.preprocessor import AISPreprocessor

def evaluate_real_data_fixed():
    print("=== SeaSeeAI Real Data Evaluation ===")
    
    # 1. Load real data for evaluation
    print("1. Loading real AIS data for evaluation...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        # Use a small different subset for evaluation
        eval_df = real_df.iloc[2000:4000]  # Next 2,000 records
        preprocessor = AISPreprocessor()
        clean_df = preprocessor.clean_data(eval_df)
        sequences, targets = preprocessor.create_sequences(clean_df, sequence_length=5, prediction_length=3)
        print(f"   Evaluation data: {len(sequences)} sequences")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Load model trained on real data
    print("2. Loading model...")
    
    # Match the model architecture used in training
    model = SimpleLSTMModel(input_size=4, hidden_size=32, output_size=4, prediction_length=3)
    try:
        model.load_state_dict(torch.load('models/lstm_real_data.pth'))
        print("   ✅ LSTM (real data) loaded")
    except Exception as e:
        print(f"   ❌ LSTM (real data) not found: {e}")
        print("   Please train the model first")
        return
    
    # 3. Evaluate model
    print("3. Evaluating on real data...")
    
    model.eval()
    with torch.no_grad():
        # Use first 50 sequences for evaluation
        test_sequences = sequences[:50]
        test_targets = targets[:50]
        
        predictions = model(torch.FloatTensor(test_sequences)).numpy()
    
    # Calculate position errors (in degrees)
    position_errors = np.sqrt((predictions[:, :, 0] - test_targets[:, :, 0])**2 + 
                             (predictions[:, :, 1] - test_targets[:, :, 1])**2)
    
    # Convert to kilometers (approx. 111 km per degree)
    position_errors_km = position_errors * 111
    
    avg_error_deg = np.mean(position_errors)
    avg_error_km = np.mean(position_errors_km)
    final_error_km = np.mean(position_errors_km[:, -1])
    
    # 4. Display results
    print("4. Real Data Evaluation Results:")
    print(f"   Average Position Error: {avg_error_km:.2f} km")
    print(f"   Final Position Error: {final_error_km:.2f} km")
    print(f"   Error in degrees: {avg_error_deg:.4f}°")
    print(f"   Tested on {len(test_sequences)} sequences")
    print(f"   Prediction horizon: {predictions.shape[1]} steps")
    
    # 5. Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training results if available
    try:
        training_img = plt.imread('real_data_training.png')
        plt.subplot(2, 2, 1)
        plt.imshow(training_img)
        plt.axis('off')
        plt.title('Training Progress')
    except:
        plt.subplot(2, 2, 1)
        plt.text(0.5, 0.5, 'Training completed!\nPlot not available', ha='center', va='center')
        plt.axis('off')
    
    # Plot 2: Error by prediction step
    plt.subplot(2, 2, 2)
    steps = range(1, predictions.shape[1] + 1)
    mean_errors_per_step = np.mean(position_errors_km, axis=0)
    
    plt.plot(steps, mean_errors_per_step, 'ro-', markersize=6)
    plt.xlabel('Prediction Step')
    plt.ylabel('Error (km)')
    plt.title('Error vs Prediction Horizon')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sample trajectory
    plt.subplot(2, 2, 3)
    sample_idx = 0
    actual_traj = test_targets[sample_idx]
    pred_traj = predictions[sample_idx]
    
    plt.plot(actual_traj[:, 1], actual_traj[:, 0], 'bo-', label='Actual', markersize=4)
    plt.plot(pred_traj[:, 1], pred_traj[:, 0], 'ro-', label='Predicted', markersize=4)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Trajectory Prediction\nReal AIS Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, 'SeaSeeAI Real Data Results:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f'Avg Error: {avg_error_km:.1f} km', fontsize=11)
    plt.text(0.1, 0.6, f'Final Error: {final_error_km:.1f} km', fontsize=11)
    plt.text(0.1, 0.5, f'Test Sequences: {len(test_sequences)}', fontsize=11)
    plt.text(0.1, 0.4, f'Real Data Points: 1M+', fontsize=11)
    plt.text(0.1, 0.3, f'Vessels: 14,006', fontsize=11)
    plt.text(0.1, 0.2, f'Memory Safe: ✅', fontsize=11, color='green')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('real_data_evaluation.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data evaluation completed!")
    print(f"   Results saved: real_data_evaluation.png")
    
    return avg_error_km, final_error_km

if __name__ == "__main__":
    evaluate_real_data_fixed()
