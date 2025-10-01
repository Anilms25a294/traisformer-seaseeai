"""
Evaluate LSTM model on real AIS data
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

def evaluate_real_data_simple():
    print("=== SeaSeeAI Real Data Evaluation ===")
    
    # 1. Load real data
    print("1. Loading real AIS data...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        # Use a different subset for evaluation
        eval_df = real_df.iloc[50000:70000]  # Different 20,000 records
        preprocessor = AISPreprocessor()
        clean_df = preprocessor.clean_data(eval_df)
        sequences, targets = preprocessor.create_sequences(clean_df)
        print(f"   Evaluation data: {len(sequences)} sequences")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Load model trained on real data
    print("2. Loading model...")
    
    model = SimpleLSTMModel(input_size=4, hidden_size=64, output_size=4, prediction_length=5)
    try:
        model.load_state_dict(torch.load('models/lstm_real_data.pth'))
        print("   ✅ LSTM (real data) loaded")
    except:
        print("   ❌ LSTM (real data) not found - train it first")
        return
    
    # 3. Evaluate model
    print("3. Evaluating on real data...")
    
    model.eval()
    with torch.no_grad():
        # Use first 100 sequences for evaluation
        test_sequences = sequences[:100]
        test_targets = targets[:100]
        
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
    
    # 5. Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training results
    try:
        training_img = plt.imread('real_data_training.png')
        plt.subplot(2, 2, 1)
        plt.imshow(training_img)
        plt.axis('off')
        plt.title('Training on Real AIS Data')
    except:
        plt.subplot(2, 2, 1)
        plt.text(0.5, 0.5, 'Training plot\nnot available', ha='center', va='center')
        plt.axis('off')
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    plt.hist(position_errors_km.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Position Error (km)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution\nReal AIS Data')
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
    plt.title('Sample Trajectory Prediction\nReal Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, 'Real AIS Data Results:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f'Avg Error: {avg_error_km:.1f} km', fontsize=11)
    plt.text(0.1, 0.6, f'Final Error: {final_error_km:.1f} km', fontsize=11)
    plt.text(0.1, 0.5, f'Test Sequences: {len(test_sequences)}', fontsize=11)
    plt.text(0.1, 0.4, f'Data Points: {len(real_df):,}', fontsize=11)
    plt.text(0.1, 0.3, f'Vessels: {real_df["mmsi"].nunique()}', fontsize=11)
    plt.text(0.1, 0.2, f'Date: 2024-12-31', fontsize=11)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('real_data_evaluation.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data evaluation completed!")
    print(f"   Results saved: real_data_evaluation.png")
    
    return avg_error_km, final_error_km

if __name__ == "__main__":
    evaluate_real_data_simple()
