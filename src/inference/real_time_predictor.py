"""
Real-time trajectory prediction for SeaSeeAI
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.traisformer import TrAISformer
from data_processing.sample_data_generator import generate_realistic_ais_data

class RealTimePredictor:
    def __init__(self, model_path, sequence_length=20, prediction_length=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Load model
        self.model, self.config = self.load_model(model_path)
        self.model.eval()
        
        # Initialize history buffer
        self.history = []
        
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = TrAISformer(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model, checkpoint['config']
    
    def add_observation(self, observation):
        """Add new AIS observation to history"""
        # observation should be [latitude, longitude, sog, cog]
        self.history.append(observation)
        
        # Keep only the most recent sequence_length observations
        if len(self.history) > self.sequence_length:
            self.history = self.history[-self.sequence_length:]
    
    def predict(self):
        """Predict future trajectory"""
        if len(self.history) < self.sequence_length:
            print(f"Need {self.sequence_length} observations, have {len(self.history)}")
            return None
        
        # Prepare input sequence
        sequence = np.array(self.history[-self.sequence_length:])
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy().squeeze()
        
        return prediction
    
    def simulate_real_time(self, vessel_data, steps=30):
        """Simulate real-time prediction on vessel data"""
        print("Starting real-time simulation...")
        
        positions = []
        actual_futures = []
        predicted_futures = []
        
        # Calculate maximum possible steps
        max_steps = len(vessel_data) - self.sequence_length - self.prediction_length
        actual_steps = min(steps, max_steps)
        
        if actual_steps <= 0:
            print(f"Not enough data for simulation. Need at least {self.sequence_length + self.prediction_length + 1} points, have {len(vessel_data)}")
            return positions, actual_futures, predicted_futures
        
        print(f"Running {actual_steps} simulation steps...")
        
        for i in range(actual_steps):
            # Get current observation
            current_obs = vessel_data[i + self.sequence_length]
            current_features = [current_obs['latitude'], current_obs['longitude'], 
                              current_obs['sog'], current_obs['cog']]
            
            # Build history
            history_slice = vessel_data[i:i + self.sequence_length]
            self.history = [[obs['latitude'], obs['longitude'], obs['sog'], obs['cog']] 
                           for obs in history_slice]
            
            # Make prediction
            prediction = self.predict()
            
            if prediction is not None:
                # Store results
                positions.append((current_obs['latitude'], current_obs['longitude']))
                
                # Actual future positions
                actual_future = vessel_data[i + self.sequence_length + 1: 
                                          i + self.sequence_length + 1 + self.prediction_length]
                actual_positions = [(obs['latitude'], obs['longitude']) for obs in actual_future]
                actual_futures.append(actual_positions)
                
                # Predicted future positions - FIXED: use 'prediction' not 'pred'
                predicted_positions = [(prediction[j, 0], prediction[j, 1]) for j in range(self.prediction_length)]
                predicted_futures.append(predicted_positions)
                
                # Print progress
                if i % 10 == 0:
                    print(f"Step {i}: Predicted {self.prediction_length} future positions")
            
            time.sleep(0.05)  # Simulate real-time delay
        
        return positions, actual_futures, predicted_futures
    
    def plot_real_time_results(self, positions, actual_futures, predicted_futures):
        """Plot real-time prediction results"""
        if not positions:
            print("No data to plot - simulation didn't generate any results")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Trajectory overview
        lats = [pos[0] for pos in positions]
        lons = [pos[1] for pos in positions]
        
        ax1.plot(lons, lats, 'b-', linewidth=2, label='Actual Path', alpha=0.7)
        if len(lons) > 0 and len(lats) > 0:
            ax1.scatter(lons[0], lats[0], color='green', s=100, label='Start', marker='o')
            ax1.scatter(lons[-1], lats[-1], color='red', s=100, label='End', marker='s')
        
        # Plot some predictions
        for i in range(0, len(positions), max(1, len(positions)//5)):
            if i < len(predicted_futures):
                pred_lats = [positions[i][0]] + [pos[0] for pos in predicted_futures[i]]
                pred_lons = [positions[i][1]] + [pos[1] for pos in predicted_futures[i]]
                ax1.plot(pred_lons, pred_lats, 'r--', alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Vessel Trajectory with Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction error over time
        errors = []
        for i in range(len(actual_futures)):
            if i < len(predicted_futures) and len(actual_futures[i]) > 0:
                # Calculate distance error for first prediction step
                actual_pos = actual_futures[i][0]  # First future position
                pred_pos = predicted_futures[i][0]  # First predicted position
                
                # Simple Euclidean distance for error (in degrees)
                error = np.sqrt((actual_pos[0] - pred_pos[0])**2 + 
                              (actual_pos[1] - pred_pos[1])**2)
                errors.append(error)
        
        if errors:
            ax2.plot(range(len(errors)), errors, 'r-', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Prediction Error (degrees)')
            ax2.set_title('Real-time Prediction Error')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(errors)
            ax2.axhline(y=mean_error, color='blue', linestyle='--', 
                       label=f'Mean Error: {mean_error:.4f}°')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No error data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Real-time Prediction Error')
        
        plt.tight_layout()
        plt.savefig('real_time_predictions.png', dpi=300, bbox_inches='tight')
        print("Real-time prediction plot saved as 'real_time_predictions.png'")
        
        return fig

def demo_real_time():
    """Demo real-time prediction"""
    print("=== SeaSeeAI Real-time Prediction Demo ===")
    
    # Generate more sample vessel data to ensure we have enough
    df = generate_realistic_ais_data(num_vessels=1, hours=72)  # Increased to 72 hours
    vessel_data = df[df['mmsi'] == df['mmsi'].iloc[0]].to_dict('records')
    
    print(f"Generated {len(vessel_data)} data points for vessel")
    
    # Initialize predictor
    predictor = RealTimePredictor('models/best_traisformer.pth')
    
    # Run real-time simulation
    positions, actual_futures, predicted_futures = predictor.simulate_real_time(vessel_data, steps=30)
    
    if positions:
        # Plot results
        predictor.plot_real_time_results(positions, actual_futures, predicted_futures)
        print("Real-time demo completed!")
    else:
        print("Real-time demo failed: insufficient data for simulation")
    
    return predictor

if __name__ == "__main__":
    demo_real_time()
