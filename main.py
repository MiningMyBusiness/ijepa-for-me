# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import multiprocessing as mp
import pprint
import yaml
import logging
import sys
import os
import glob

from src.utils.distributed import init_distributed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    from src.train import main as app_main
    app_main(args=params)

def main():
    args = parser.parse_args()
    
    logger.info(f"Starting training with config file: {args.fname}")
    logger.info(f"Using devices: {args.devices}")
    
    # Check if config file exists
    if not os.path.exists(args.fname):
        logger.error(f"Config file not found: {args.fname}")
        return
    
    try:
        # Load config file
        logger.info(f"Loading configuration from {args.fname}")
        with open(args.fname, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
        # Log some key configuration parameters
        logger.info(f"Model: {config['meta']['model_name']}")
        logger.info(f"Batch size: {config['data']['batch_size']}")
        logger.info(f"Number of epochs: {config['optimization']['epochs']}")
        
        # Validate image folder path
        image_folder = config['data']['image_folder']
        if image_folder is None:
            logger.error("image_folder is not specified in the config!")
        elif not os.path.exists(image_folder):
            logger.error(f"image_folder path does not exist: {image_folder}")
        else:
            logger.info(f"image_folder path exists: {image_folder}")
            # Count number of image files
            image_count = 0
            image_dirs = glob.glob(os.path.join(image_folder, 'pcid_*_ci'))
            for this_dir in image_dirs:
                image_count += len(glob.glob(os.path.join(this_dir, '*.jpg')))
            logger.info(f"Found {image_count} image files in image_folder")
        
        # Import train module
        logger.info("Importing training module")
        from src.train import main as train_main
        
        # Call the training function
        logger.info("Starting training process")
        train_main(config)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()