# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import glob
from torchvision import transforms

_GLOBAL_SEED = 0
logger = getLogger()


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    use_video_dataset=True,
):
    if not use_video_dataset:
        dataset = ImageNet(
            root=root_path,
            image_folder=image_folder,
            transform=transform,
            train=training,
            copy_data=copy_data,
            index_targets=False)
        if subset_file is not None:
            dataset = ImageNetSubset(dataset, subset_file)
        logger.info('ImageNet dataset created')
    else:
        # Check if image_folder is a valid path
        if image_folder is None:
            logger.error("image_folder is None! Please provide a valid path.")
            image_folder = "."  # Default to current directory
        
        if not os.path.exists(image_folder):
            logger.error(f"image_folder path does not exist: {image_folder}")
            
        dataset = VideoFrameDataset(
            video_dirs=image_folder,
            transform=transform,
        )
        logger.info(f'Video frame dataset created with {len(dataset)} images')
    
    # Check if dataset is empty
    if len(dataset) == 0:
        logger.error("Dataset is empty! No images found.")
        # Create a dummy dataset with a single black image to prevent crashes
        from torch.utils.data import TensorDataset
        import torch
        dummy_img = torch.zeros(3, 224, 224)
        dataset = TensorDataset(dummy_img.unsqueeze(0), torch.tensor([0]))
        logger.warning("Created dummy dataset with one black image to prevent crashes")
    
    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    logger.info(f'Data loader created with {len(data_loader)} batches')

    return dataset, data_loader, dist_sampler


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path


class VideoFrameDataset(Dataset):
    def __init__(self, video_dirs, transform=None):
        """
        Dataset for loading video frames as images
        
        Args:
            video_dirs: Path to directory containing video frames or images
            transform: Torchvision transforms to apply to images
        """
        self.transform = transform
        self.image_paths = []
        
        # Check if video_dirs is a string (single directory) or list of directories
        if isinstance(video_dirs, str):
            video_dirs = [video_dirs]
            
        # Collect all image files from the directories
        for directory in video_dirs:
            if not os.path.exists(directory):
                logger.error(f"Directory does not exist: {directory}")
                continue
                
            logger.info(f"Loading images from directory: {directory}")
            
            # Get all files in the directory and its subdirectories
            image_dirs = glob.glob(os.path.join(directory, 'pcid_*_ci'))
            for this_dir in image_dirs:
                this_image_paths = glob.glob(os.path.join(this_dir, '*.jpg'))
                self.image_paths.extend(this_image_paths)
        
        # Log the number of images found
        logger.info(f"Found {len(self.image_paths)} images in the dataset")
        
        if len(self.image_paths) == 0:
            logger.warning("No images found in the specified directories!")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load image using PIL
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms if specified
            if self.transform is not None:
                img = self.transform(img)
                
            # Return image with a dummy target (0) for compatibility with ImageNet dataset
            return img, 0
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder in case of error
            if self.transform is not None:
                # Create a black image of the expected size
                img = Image.new('RGB', (224, 224), color=0)
                return self.transform(img), 0
            return None, 0