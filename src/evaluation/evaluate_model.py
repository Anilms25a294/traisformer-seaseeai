"""
Evaluate the trained model on real AIS data
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def evaluate_predictions():
    print("=== Model Evaluation ===")
    
    # Load real data
    df = pd.read_csv("AIS_2024_12_311.csv", nrows=1000)
    data = df[['LAT', 'LON', 'SOG', 'COG']].values
    
    # Create test sequences (different from training)
    test_sequences = []
    test_targets = []
    for i in range(800, len(data) - 8):  # Use later data for testing
        test_sequences.append(data[i:i+5])
        test_targets.append(data[i+5:i+8])
    
    X_test = torch.FloatTensor(np.array(test_sequences))
    y_test = torch.FloatTensor(np.array(test_targets))
    
    print(f"Test data: {X_test.shape} sequences")
    
    # Calculate baseline (naive prediction: repeat last position)
    naive_pred = X_test[:, -1:, :].repeat(1, 3, 1)
    naive_error = torch.mean((naive_pred - y_test) ** 2)
    print(f"Naive baseline MSE: {naive_error.item():.6f}")
    
    # Load and test trained model if available
    try:
        from models.baseline import SimpleLSTMModel
        model = SimpleLSTMModel(input_size=4, prediction_length=3)
        # You would load trained weights here: model.load_state_dict(...)
        
        with torch.no_grad():
            predictions = model(X_test)
            model_error = torch.mean((predictions - y_test) ** 2)
            print(f"Model MSE: {model_error.item():.6f}")
            
            if model_error < naive_error:
                print("✅ Model beats naive baseline!")
            else:
                print("⚠️  Model needs improvement")
                
    except Exception as e:
        print(f"Model evaluation skipped: {e}")
    
    return naive_error.item()

if __name__ == "__main__":
    evaluate_predictions()
