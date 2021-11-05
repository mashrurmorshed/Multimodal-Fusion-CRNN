from argparse import ArgumentParser
from config_parser import get_config

from utils.loss import LabelSmoothingLoss
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import train
from utils.load_DHG import get_loaders, init_cache
from utils.misc import seed_everything, count_params, get_model

import torch
from torch import nn
import numpy as np
import wandb

import os
import yaml
import random
import time


def training_pipeline(config, cache = None):
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict): Dict containing various settings for the training run.
        cache (dict): Cache containing the pre-loaded dataset.
    """

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    
    ######################################
    # initialize training items
    ######################################

    # data
    loaders = get_loaders(config, cache)

    # model
    model = get_model(config["hparams"]["model"])
    model = model.to(config["hparams"]["device"])
    print(f"Created model with {count_params(model)} parameters.")

    # loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(num_classes=config["hparams"]["model"]["num_classes"], smoothing=config["hparams"]["l_smooth"])
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = get_optimizer(model, config["hparams"])
    
    # lr scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }

    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(loaders["train"]) * config["hparams"]["scheduler"]["n_warmup"])

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(loaders["train"]) * max(1, (config["hparams"]["n_epochs"] - config["hparams"]["scheduler"]["n_warmup"]))
        schedulers["scheduler"] = get_scheduler(optimizer, config["hparams"]["scheduler"]["scheduler_type"], total_iters)
    

    ######################################
    # Training Run
    ######################################

    print("Initiating training.")
    train(model, optimizer, criterion, loaders["train"], loaders["val"], schedulers, config)



def main(args):
    config = get_config(args.conf)

    seed_everything(config["hparams"]["seed"])

    #################################
    # single time global caching
    #################################
    
    data_list = np.loadtxt(config["data_list_path"], np.int32)

    cache = None
    if config["exp"]["cache"]:
        cache = init_cache(
            data_list,
            config["data_root"],
            config["hparams"]["model"]["T"],
            config["hparams"]["model"]["D"],
            config["hparams"]["transforms"]["train"],
            config["exp"]["n_cache_workers"]
        )

    #################################
    # leave one out cross validation
    #################################
    
    subjects = np.unique(data_list[:, 2]).tolist()

    for sub in subjects:
        config["exp"]["val_sub"] = sub
        config["exp"]["exp_name"] = f"sub_{sub}"

        if config["exp"]["wandb"]:
            if config["exp"]["wandb_api_key"] is not None:
                with open(config["exp"]["wandb_api_key"], "r") as f:
                    os.environ["WANDB_API_KEY"] = f.read()

            elif os.environ.get("WANDB_API_KEY", False):
                print(f"Found API key from env variable.")

            else:
                wandb.login()
            

            with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"], tags=config["exp"]["tags"], group=config["exp"]["group"]):
                training_pipeline(config, cache)
        
        else:
            training_pipeline(config, cache)



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)