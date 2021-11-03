"""Miscellaneous helper functions."""

import torch
from torch import nn, optim
import numpy as np
import random
import os
import wandb
from models.feature_fusion import FeatureFusionNet
from models.score_fusion import ScoreFusionNet

def seed_everything(seed: str) -> None:
    """Set manual seed.

    Args:
        seed (int): Supplied seed.
    """
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set seed {seed}')


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.

    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.

    Returns:
        int: Parameter count.
    """
    
    return sum(map(lambda p: p.data.numel(), model.parameters()))


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.

    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.

    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict, step: int, config: dict) -> None:
    """Handles logging for metric tracking server, local disk and stdout.

    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """
    # send logs to wandb tracking server
    if config["exp"]["wandb"]:
        wandb.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join([f"{k}: {v}" for k, v in log_dict.items()])
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    
    # write logs to disk
    with open(log_file, "a+") as f:
        f.write(log_message + "\n")

    # show logs in stdout
    print(log_message)


def get_model(model_config):
    if model_config["name"] == "feature_fusion":
        fusion_model = FeatureFusionNet
    elif model_config["name"] == "score_fusion":
        fusion_model = ScoreFusionNet
    else:
        raise ValueError(f"Invalid model_name {model_config['name']}.")

    return fusion_model(**model_config)

def save_model(epoch: int, save_path: str, net: nn.Module, optimizer : optim.Optimizer = None, log_file : str = None) -> None:
    """Saves checkpoint.

    Args:
        epoch (int): Current epoch.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path}."

    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")
    print(log_message)