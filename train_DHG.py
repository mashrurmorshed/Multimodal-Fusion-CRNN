from argparse import ArgumentParser
from config_parser import get_config

from utils.loss import LabelSmoothingLoss
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import train, evaluate_stats
from utils.load_DHG import get_loaders, init_cache
from utils.misc import seed_everything, count_params, get_model, log
from utils.plotcm import make_confusion_matrix

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
    criterion = LabelSmoothingLoss(
        num_classes=config["hparams"]["model"]["num_classes"],
        smoothing=config["hparams"]["l_smooth"],
        logits=False if config["hparams"]["model"]["type"]=="decision_fusion" else True
    )

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


    #####################################
    # Get statistics
    #####################################

    stats = None
    if config["exp"]["get_stats"]:
        # restore ckpt
        ckpt_path = os.path.join(config["exp"]["save_dir"], "best.pth")
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        print("Successfully restored state.")

        print("Getting fine/coarse statistics.")
        stats = evaluate_stats(model, loaders["val"], config["hparams"]["device"])
        log_dict = {
            "fine": stats["fine"],
            "coarse": stats["coarse"]
        }
        step = config["hparams"]["n_epochs"] * len(loaders["train"])
        log(log_dict, step, config)
    
    return stats


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
            config["hparams"]["transforms"],
            config["exp"]["n_cache_workers"]
        )

    #################################
    # leave one out cross validation
    #################################
    
    subjects = np.unique(data_list[:, 2]).tolist()
    all_labels, all_preds = [], []
    
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
                stats = training_pipeline(config, cache)
        
        else:
            stats = training_pipeline(config, cache)

        if stats != None:
            all_labels.append(stats["labels"])
            all_preds.append(stats["preds"])
        
    if config["exp"]["get_stats"]:
        all_labels = np.hstack(all_labels).ravel()
        all_preds = np.hstack(all_preds).ravel()
        make_confusion_matrix(all_labels, all_preds, config["exp"]["cm_path"])
    
    



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)