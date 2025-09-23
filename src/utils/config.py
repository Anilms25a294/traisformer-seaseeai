"""
Configuration settings for SeaSeeAI project
"""

class SeaSeeConfig:
    # Data parameters
    SEQUENCE_LENGTH = 20      # Past time steps to use for prediction
    PREDICTION_LENGTH = 10    # Future time steps to predict
    BATCH_SIZE = 32
    
    # Model parameters
    D_MODEL = 128            # Transformer embedding dimension
    NHEAD = 8                # Number of attention heads
    NUM_LAYERS = 4           # Number of transformer layers
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # File paths
    RAW_DATA_PATH = "data/raw/"
    PROCESSED_DATA_PATH = "data/processed/"
    
    @staticmethod
    def get_features():
        """Features to use for prediction"""
        return ['latitude', 'longitude', 'sog', 'cog']  # SOG = Speed Over Ground, COG = Course Over Ground
