"""
Evaluate the working SeaSeeAI model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing.preprocessor import AISPreprocessor
from data_processing.sample_data_generator import generate_realistic_ais_data

# Same model as in working_train.py
class WorkingModel(torch.nn.Module):
    def __init__(self, seq_len=10, pred_len=5, features=4):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(seq_len * features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, pred_len * features)
        )
        self.pred_len = pred_len
        self.features = features
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        out = self.fc(x)
        return out.reshape(batch_size, self.pred_len, self.features)

def evaluate_working_model():
    print("=== SeaSeeAI Working Model Evaluation ===")
    
    # 1. Load trained model
    try:
        model = WorkingModel()
        model.load_state_dict(torch.load('models/working_model.pth'))
        model.eval()
        print("✅ Working model loaded")
    except Exception as e:
        print(f"❌ No working model found: {e}")
        print("Please run working_train.py first")
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
    
    # Position error (in degrees)
    position_errors = np.sqrt((predictions[:, :, 0] - actual[:, :, 0])**2 + 
                             (predictions[:, :, 1] - actual[:, :, 1])**2)
    
    avg_position_error = np.mean(position_errors)
    final_position_error = np.mean(position_errors[:, -1])
    
    print("📊 Evaluation Results:")
    print(f"   Average Position Error: {avg_position_error:.4f} degrees")
    print(f"   Final Position Error: {final_position_error:.4f} degrees")
    print(f"   Tested on {len(actual)} sequences")
    
    # 5. Convert to meaningful units (approx. 111 km per degree)
    avg_error_km = avg_position_error * 111
    final_error_km = final_position_error * 111
    
    print(f"   Average Error: {avg_error_km:.2f} km")
    print(f"   Final Error: {final_error_km:.2f} km")
    
    # 6. Create evaluation plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training results
    try:
        training_img = plt.imread('working_training.png')
        plt.subplot(2, 2, 1)
        plt.imshow(training_img)
        plt.axis('off')
        plt.title('Training Progress')
    except:
        plt.subplot(2, 2, 1)
        plt.text(0.5, 0.5, 'Training plot\nnot available', ha='center', va='center')
        plt.axis('off')
    
    # Plot 2: Position errors over time
    plt.subplot(2, 2, 2)
    time_steps = range(1, 6)
    mean_errors = np.mean(position_errors, axis=0)
    
    plt.plot(time_steps, mean_errors, 'bo-', markersize=6, label='Position Error')
    plt.title('Error vs Prediction Horizon')
    plt.xlabel('Steps Ahead')
    plt.ylabel('Error (degrees)')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Sample trajectory
    plt.subplot(2, 2, 3)
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
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Avg Error: {avg_position_error:.4f}°', fontsize=12)
    plt.text(0.1, 0.7, f'Final Error: {final_position_error:.4f}°', fontsize=12)
    plt.text(0.1, 0.6, f'Avg Distance: {avg_error_km:.1f} km', fontsize=12)
    plt.text(0.1, 0.5, f'Final Distance: {final_error_km:.1f} km', fontsize=12)
    plt.text(0.1, 0.4, f'Test Sequences: {len(actual)}', fontsize=12)
    plt.text(0.1, 0.3, 'Model: Working Model', fontsize=12)
    plt.axis('off')
    plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('working_model_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ Evaluation plot saved as 'working_model_evaluation.png'")
    
    return avg_position_error, final_position_error

if __name__ == "__main__":
    evaluate_working_model()
