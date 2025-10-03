"""
Test script to verify all fixes are working
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_lstm_model():
    """Test the fixed LSTM model"""
    print("Testing LSTM model...")
    try:
        from models.simple_lstm_model import SimpleLSTMModel
        
        # Test model shape
        model = SimpleLSTMModel(input_size=4, hidden_size=64, output_size=4, prediction_length=5)
        sample_input = torch.randn(2, 10, 4)
        output = model(sample_input)
        
        print(f"✅ LSTM Model: Input {sample_input.shape} -> Output {output.shape}")
        assert output.shape == (2, 5, 4), f"Expected (2, 5, 4), got {output.shape}"
        print("✅ LSTM model output shape is correct!")
        
    except Exception as e:
        print(f"❌ LSTM model test failed: {e}")
        return False
    return True

def test_transformer_model():
    """Test the Transformer model"""
    print("Testing Transformer model...")
    try:
        from models.traisformer import TrAISformer
        
        # Test model shape
        model = TrAISformer(input_size=4, prediction_length=5)
        sample_input = torch.randn(2, 20, 4)
        output = model(sample_input)
        
        print(f"✅ Transformer Model: Input {sample_input.shape} -> Output {output.shape}")
        assert output.shape == (2, 5, 4), f"Expected (2, 5, 4), got {output.shape}"
        print("✅ Transformer model output shape is correct!")
        
    except Exception as e:
        print(f"❌ Transformer model test failed: {e}")
        return False
    return True

def test_data_generation():
    """Test data generation"""
    print("Testing data generation...")
    try:
        from data_processing.sample_data_generator import generate_realistic_ais_data
        
        df = generate_realistic_ais_data(num_vessels=2, hours=24)
        print(f"✅ Generated {len(df)} data points")
        assert len(df) > 0, "No data generated"
        print("✅ Data generation works!")
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False
    return True

def main():
    """Run all tests"""
    print("Running SeaSeeAI Fix Verification Tests...")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 3
    
    if test_lstm_model():
        tests_passed += 1
    
    if test_transformer_model():
        tests_passed += 1
        
    if test_data_generation():
        tests_passed += 1
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! You can now run the training pipeline.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
