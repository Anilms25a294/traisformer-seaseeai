"""
Evaluate models on real AIS data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline import SimpleLSTMModel
from models.transformer import TrAISformer
from data_processing.preprocessor import AISPreprocessor

def evaluate_real_data():
    print("=== SeaSeeAI Real Data Evaluation ===")
    
    # 1. Load real data
    print("1. Loading real AIS data...")
    try:
        real_df = pd.read_csv("data/real_ais/processed_ais_data.csv")
        preprocessor = AISPreprocessor()
        clean_df = preprocessor.clean_data(real_df)
        sequences, targets = preprocessor.create_sequences(clean_df)
        print(f"   Evaluation data: {len(sequences)} sequences")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Load models trained on real data
    print("2. Loading models...")
    
    # LSTM trained on real data
    lstm_real = SimpleLSTMModel(input_size=4, hidden_size=64, output_size=4, prediction_length=5)
    try:
        lstm_real.load_state_dict(torch.load('models/lstm_real_data.pth'))
        print("   ✅ LSTM (real data) loaded")
    except:
        print("   ❌ LSTM (real data) not found")
        return
    
    # Transformer trained on real data
    transformer_real = TrAISformer(input_size=4, d_model=128, nhead=8, num_layers=4, prediction_length=5)
    try:
        transformer_real.load_state_dict(torch.load('models/transformer_real_data.pth'))
        print("   ✅ Transformer (real data) loaded")
    except:
        print("   ❌ Transformer (real data) not found")
        return
    
    # 3. Evaluate models
    print("3. Evaluating on real data...")
    
    def calculate_metrics(model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            predictions = model(torch.FloatTensor(X_test)).numpy()
        
        # Position errors (in degrees)
        position_errors = np.sqrt((predictions[:, :, 0] - y_test[:, :, 0])**2 + 
                                 (predictions[:, :, 1] - y_test[:, :, 1])**2)
        
        # Convert to kilometers (approx. 111 km per degree)
        position_errors_km = position_errors * 111
        
        avg_error_deg = np.mean(position_errors)
        avg_error_km = np.mean(position_errors_km)
        final_error_km = np.mean(position_errors_km[:, -1])
        
        return avg_error_deg, avg_error_km, final_error_km
    
    # Use first 50 sequences for evaluation
    test_sequences = sequences[:50]
    test_targets = targets[:50]
    
    lstm_avg_deg, lstm_avg_km, lstm_final_km = calculate_metrics(lstm_real, test_sequences, test_targets)
    transformer_avg_deg, transformer_avg_km, transformer_final_km = calculate_metrics(transformer_real, test_sequences, test_targets)
    
    # 4. Display results
    print("4. Real Data Evaluation Results:")
    print(f"   LSTM - Average Error: {lstm_avg_km:.2f} km, Final Error: {lstm_final_km:.2f} km")
    print(f"   Transformer - Average Error: {transformer_avg_km:.2f} km, Final Error: {transformer_final_km:.2f} km")
    
    # 5. Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Error comparison
    plt.subplot(2, 3, 1)
    models = ['LSTM', 'Transformer']
    avg_errors = [lstm_avg_km, transformer_avg_km]
    final_errors = [lstm_final_km, transformer_final_km]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, avg_errors, width, label='Average Error', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, final_errors, width, label='Final Error', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Model')
    plt.ylabel('Error (km)')
    plt.title('Real Data Performance\n(Lower is Better)')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Data overview
    plt.subplot(2, 3, 2)
    plt.text(0.1, 0.8, f'Real AIS Data Summary:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.6, f'Total Records: {len(real_df):,}', fontsize=10)
    plt.text(0.1, 0.5, f'Unique Vessels: {real_df["mmsi"].nunique()}', fontsize=10)
    plt.text(0.1, 0.4, f'Evaluation Sequences: {len(test_sequences)}', fontsize=10)
    plt.text(0.1, 0.3, f'Data File: AIS_2024_12_311.csv', fontsize=10)
    plt.axis('off')
    
    # Plot 3: Sample trajectory from real data
    plt.subplot(2, 3, 3)
    # Get a sample vessel trajectory
    sample_vessel = real_df['mmsi'].iloc[0]
    vessel_data = real_df[real_df['mmsi'] == sample_vessel].head(20)
    
    plt.plot(vessel_data['longitude'], vessel_data['latitude'], 'bo-', alpha=0.7, markersize=3)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Sample Vessel Trajectory\nMMSI: {sample_vessel}')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement
    plt.subplot(2, 3, 4)
    improvement = ((lstm_avg_km - transformer_avg_km) / lstm_avg_km) * 100
    colors = ['lightgreen' if improvement > 0 else 'lightcoral']
    
    plt.bar(['Transformer vs LSTM'], [improvement], color=colors, alpha=0.8)
    plt.ylabel('Improvement (%)')
    plt.title('Performance Improvement\n(Positive = Better)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Geographic coverage
    plt.subplot(2, 3, 5)
    plt.scatter(real_df['longitude'], real_df['latitude'], alpha=0.1, s=1, color='blue')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Coverage\nAll Vessel Positions')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Speed distribution
    plt.subplot(2, 3, 6)
    plt.hist(real_df['sog'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Speed (knots)')
    plt.ylabel('Frequency')
    plt.title('Vessel Speed Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_evaluation.png', dpi=150, bbox_inches='tight')
    
    print("✅ Real data evaluation completed!")
    print(f"   Results saved: real_data_evaluation.png")
    print(f"   Key Insight: Transformer shows {improvement:.1f}% improvement over LSTM")
    
    return lstm_avg_km, transformer_avg_km

if __name__ == "__main__":
    evaluate_real_data()
