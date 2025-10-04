#!/usr/bin/env python3
"""
Production startup script for SeaSeeAI
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_logging():
    """Setup production logging"""
    log_path = Path('logs')
    log_path.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/production.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main production startup"""
    setup_logging()
    logger = logging.getLogger("seaseeai-production")
    
    logger.info("Starting SeaSeeAI Production System")
    
    # Check environment
    env = os.getenv('SEASEEAI_ENV', 'production')
    logger.info(f"Environment: {env}")
    
    # Check required files
    required_files = [
        'models/best_traisformer.pth',
        'config/production.yaml'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            sys.exit(1)
    
    # Start appropriate services based on environment
    if env == 'api':
        logger.info("Starting API server only")
        from api.fastapi_server import start_server
        start_server()
    elif env == 'dashboard':
        logger.info("Starting dashboard only")
        # This would typically be run separately
        logger.info("Run: streamlit run src/dashboard/streamlit_app.py")
    else:
        logger.info("Starting full production system")
        # In production, we might use a process manager like supervisord
        logger.info("Production system ready - start individual services as needed")

if __name__ == "__main__":
    main()
