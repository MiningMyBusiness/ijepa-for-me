# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import logging
import traceback

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed(port=40112, rank_and_world_size=(None, None)):
    logger.info("Initializing distributed process group")
    
    if dist.is_available() and dist.is_initialized():
        logger.info("Distributed already initialized")
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            logger.info("Trying to get SLURM environment variables")
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
            logger.info(f"SLURM vars set: world_size={world_size}, rank={rank}, master_addr={os.environ['MASTER_ADDR']}")
        except Exception as e:
            logger.info(f'SLURM vars not set (distributed training not available): {str(e)}')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        logger.info(f"Initializing process group with backend=nccl, world_size={world_size}, rank={rank}, port={port}")
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
        logger.info("Process group initialized successfully")
    except Exception as e:
        world_size, rank = 1, 0
        logger.error(f'Distributed training not available: {str(e)}')
        logger.error(traceback.format_exc())

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        if dist.is_available() and dist.is_initialized():
            output = input_.clone()
            dist.all_reduce(output)
            return output
        else:
            return input_  # Just return the input if distributed is not available

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_available() and dist.is_initialized():
            output = grad_output.clone()
            dist.all_reduce(output)
            return output
        else:
            return grad_output  # Just return the gradient if distributed is not available