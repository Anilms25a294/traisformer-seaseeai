import sys
import os

# Add the src directory to path
sys.path.insert(0, 'src')

try:
    from models.baseline import SimpleLSTMModel
    import torch
    
    print("✅ Successfully imported model!")
    
    # Test the model
    model = SimpleLSTMModel(input_size=4, prediction_length=3)
    test_input = torch.randn(2, 5, 4)
    test_output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Expected: (2, 3, 4)")
    
    if test_output.shape == (2, 3, 4):
        print("✅ Model test passed!")
    else:
        print("❌ Model test failed!")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")
