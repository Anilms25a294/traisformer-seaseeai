"""
Evaluate the trained SeaSeeAI model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.baseline import SimpleLSTMModel
    from data_processing.preprocessor import AISPreprocessor
    from data_processing.sample_data_generator import generate_realistic_ais_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are available")
    sys.exit(1)

def evaluate_model():
    print("=== SeaSeeAI Model Evaluation ===")
    
    # 1. Load trained model
    try:
        model = SimpleLSTMModel(input_size=4, hidden_size=64, output_size=4, prediction_length=5)
        model.load_state_dict(torch.load('models/baseline_lstm.pth'))
        model.eval()
        print("Trained model loaded")
    except Exception as e:
        print(f"No trained model found: {e}")
        print("Please train the model first")
        return None, None
    
    # 2. Generate test data
    print("Generating test data...")
    df = generate_realistic_ais_data(num_vessels=3, hours=24)
    preprocessor = AISPreprocessor()
    clean_df = preprocessor.clean_data(df)
    sequences, targets = preprocessor.create_sequences(clean_df)
    
    # 3. Make predictions
    with torch.no_grad():
        test_input = torch.FloatTensor(sequences[:10])
        predictions = model(test_input).numpy()
    
    # 4. Calculate metrics
    actual = targets[:10]
    
    # Position error
    position_errors = np.sqrt((predictions[:, :, 0] - actual[:, :, 0])**2 + 
                             (predictions[:, :, 1] - actual[:, :, 1])**2)
    
    avg_position_error = np.mean(position_errors)
    final_position_error = np.mean(position_errors[:, -1])
    
    print("Evaluation Results:")
    print(f"   Average Position Error: {avg_position_error:.6f} degrees")
    print(f"   Final Position Error: {final_position_error:.6f} degrees")
    print(f"   Tested on {len(actual)} sequences")
    
    # 5. Create evaluation plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Position errors
    plt.subplot(2, 2, 1)
    time_steps = range(1, 6)
    mean_errors = np.mean(position_errors, axis=0)
    
    plt.plot(time_steps, mean_errors, 'bo-', label='Mean Error')
    plt.title('Position Error vs Prediction Horizon')
    plt.xlabel('Prediction Steps Ahead')
    plt.ylabel('Error (degrees)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Sample trajectory
    plt.subplot(2, 2, 2)
    sample_idx = 0
    actual_traj = actual[sample_idx]
    pred_traj = predictions[sample_idx]
    
    plt.plot(actual_traj[:, 1], actual_traj[:, 0], 'bo-', label='Actual', markersize=4)
    plt.plot(pred_traj[:, 1], pred_traj[:, 0], 'ro-', label='Predicted', markersize=4)
    plt.title('Sample Trajectory Prediction')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Speed prediction
    plt.subplot(2, 2, 3)
    speed_actual = actual[sample_idx, :, 2]
    speed_pred = predictions[sample_idx, :, 2]
    
    plt.plot(time_steps, speed_actual, 'bo-', label='Actual Speed')
    plt.plot(time_steps, speed_pred, 'ro-', label='Predicted Speed')
    plt.title('Speed Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Speed (knots)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Training summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Avg Position Error: {avg_position_error:.4f}', fontsize=12)
    plt.text(0.1, 0.6, f'Final Position Error: {final_position_error:.4f}', fontsize=12)
    plt.text(0.1, 0.4, f'Test Sequences: {len(actual)}', fontsize=12)
    plt.text(0.1, 0.2, 'Model: LSTM Baseline', fontsize=12)
    plt.axis('off')
    plt.title('Evaluation Summary')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    print("Evaluation plot saved as 'model_evaluation.png'")
    
    return avg_position_error, final_position_error

if __name__ == "__main__":
    evaluate_model()
