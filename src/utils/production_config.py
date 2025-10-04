"""
Production configuration management for SeaSeeAI
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ProductionConfig:
    """Production configuration for SeaSeeAI"""
    
    # Model configuration
    model_checkpoint: str = "models/best_traisformer.pth"
    sequence_length: int = 20
    prediction_length: int = 5
    max_batch_size: int = 32
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 300
    
    # Database configuration
    database_url: str = "sqlite:///seaseeai.db"
    redis_url: str = "redis://localhost:6379"
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/seaseeai.log"
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Performance optimization
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    model_warmup: bool = True
    
    # Security
    api_key_required: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/production.yaml"):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        else:
            # Return default config
            return cls()
    
    def to_yaml(self, config_path: str = "config/production.yaml"):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'model_checkpoint': self.model_checkpoint,
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'max_batch_size': self.max_batch_size,
            'api_host': self.api_host,
            'api_port': self.api_port,
            'api_workers': self.api_workers,
            'api_timeout': self.api_timeout,
            'database_url': self.database_url,
            'redis_url': self.redis_url,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'enable_metrics': self.enable_metrics,
            'metrics_port': self.metrics_port,
            'health_check_interval': self.health_check_interval,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'model_warmup': self.model_warmup,
            'api_key_required': self.api_key_required,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_window': self.rate_limit_window
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self):
        """Validate configuration"""
        assert Path(self.model_checkpoint).exists(), f"Model checkpoint not found: {self.model_checkpoint}"
        assert self.sequence_length > 0, "Sequence length must be positive"
        assert self.prediction_length > 0, "Prediction length must be positive"
        assert self.max_batch_size > 0, "Max batch size must be positive"
        assert 0 < self.api_port < 65536, "API port must be valid"
        
        # Create log directory
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        return True

# Default production configuration
def get_default_config():
    """Get default production configuration"""
    return ProductionConfig()

# Environment-based configuration
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('SEASEEAI_ENV', 'production')
    
    if env == 'development':
        config = ProductionConfig()
        config.log_level = "DEBUG"
        config.enable_metrics = False
    elif env == 'testing':
        config = ProductionConfig()
        config.database_url = "sqlite:///test_seaseeai.db"
        config.log_level = "WARNING"
    else:  # production
        config_path = os.getenv('SEASEEAI_CONFIG', 'config/production.yaml')
        config = ProductionConfig.from_yaml(config_path)
    
    return config

if __name__ == "__main__":
    # Create default configuration file
    config = get_default_config()
    config.to_yaml()
    print("Default production configuration created at 'config/production.yaml'")
