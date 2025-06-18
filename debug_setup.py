import os
import sys
import yaml
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    logger.info("Checking environment setup...")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # Check config file
    config_path = "configs/in1k_vith14_ep300.yaml"
    if os.path.exists(config_path):
        logger.info(f"Config file found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Config file loaded successfully")
            logger.info(f"Model: {config['meta']['model_name']}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    else:
        logger.error(f"Config file not found: {config_path}")
    
    # Check data path
    if 'config' in locals():
        data_path = config['data']['root_path']
        logger.info(f"Data path from config: {data_path}")
        if os.path.exists(data_path):
            logger.info("Data path exists")
        else:
            logger.error(f"Data path does not exist: {data_path}")
    
    logger.info("Environment check completed")

if __name__ == "__main__":
    check_environment()
