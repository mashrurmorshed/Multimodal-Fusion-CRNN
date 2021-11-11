import torch
import numpy as np
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from utils.misc import log, calc_step, save_model
import os
import time
from tqdm import tqdm


FINE = np.array([0, 2, 3, 4, 5])
COARSE = np.array([1, 6, 7, 8, 9, 10, 11, 12, 13])


def train_single_batch(net: nn.Module, joints: torch.Tensor, images: torch.Tensor, targets: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable, device: torch.device) -> Tuple[float, int]:
    """Performs a single training step.

    Args:
        net (nn.Module): Model instance.
        data (torch.Tensor): Data tensor, of shape (batch_size, dim_1, ... , dim_N).
        targets (torch.Tensor): Target tensor, of shape (batch_size).
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        device (torch.device): Device.

    Returns:
        float: Loss scalar.
        int: Number of correct preds.
    """

    joints, images, targets = joints.to(device), images.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = net((joints, images))
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    correct = outputs.max(1)[1].eq(targets).sum()
    return loss.item(), correct.item()


@torch.no_grad()
def evaluate(net: nn.Module, criterion: Callable, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Performs inference.

    Args:
        net (nn.Module): Model instance.
        criterion (Callable): Loss function.
        dataloader (DataLoader): Test or validation loader.
        device (torch.device): Device.

    Returns:
        accuracy (float): Accuracy.
        float: Loss scalar.
    """

    net.eval()
    correct = 0
    running_loss = 0.0


    for joints, images, targets in tqdm(dataloader):
        joints, images, targets = joints.to(device), images.to(device), targets.to(device)
        out = net((joints, images))
        _, preds = out.max(1)
        correct += preds.eq(targets).sum().item()
        loss = criterion(out, targets)
        running_loss += loss.item()


    net.train()
    accuracy = correct / len(dataloader.dataset)
    return accuracy, running_loss / len(dataloader)


@torch.no_grad()
def evaluate_stats(net: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Performs inference and gets fine/coarse grain statistics.

    Args:
        net (nn.Module): Model instance.
        dataloader (DataLoader): Test or validation loader.
        device (torch.device): Device.

    Returns:
        stats (dict): Dict containing stats.
    """

    net.eval()

    stats = {"preds": [], "labels": []}

    for joints, images, targets in tqdm(dataloader):
        joints, images, targets = joints.to(device), images.to(device), targets.to(device)
        out = net((joints, images))
        _, preds = out.max(1)

        stats["preds"].append(preds.cpu().numpy())
        stats["labels"].append(targets.cpu().numpy())

    #####################################
    # accuracy for fine and coarse grain
    #####################################
    stats["preds"], stats["labels"] = np.hstack(stats["preds"]), np.hstack(stats["labels"])

    fine_idx = np.isin(stats["labels"], FINE)
    coarse_idx = np.isin(stats["labels"], COARSE)

    grain_acc_fn = lambda a: (stats["preds"][a] == stats["labels"][a]).sum() / a.sum()
    stats["fine"] = grain_acc_fn(fine_idx)
    stats["coarse"] = grain_acc_fn(coarse_idx)

    net.train()
    return stats


def train(net: nn.Module, optimizer: optim.Optimizer, criterion: Callable, trainloader: DataLoader, valloader: DataLoader, schedulers: dict, config: dict) -> None:
    """Trains model.

    Args:
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        schedulers (dict): Dict containing schedulers.
        config (dict): Config dict.
    """
    
    best_acc = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    
    ############################
    # start training
    ############################
    net.train()
    
    for epoch in range(1, config["hparams"]["n_epochs"] + 1):
        t0 = time.time()
        running_loss = 0.0
        correct = 0

        for batch_index, (joints, images, targets) in enumerate(trainloader):
            step = calc_step(epoch, n_batches, batch_index)

            # schedulers
            
            if schedulers["warmup"] is not None and epoch <= config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()
            
            elif schedulers["scheduler"] is not None and epoch > config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################
            loss, corr = train_single_batch(net, joints, images, targets, optimizer, criterion, device)
            running_loss += loss
            correct += corr

            if not batch_index % config["exp"]["log_freq"]:       
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                log(log_dict, step, config)
        
        #######################
        # epoch complete
        #######################
        train_acc = correct / (len(trainloader.dataset))
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": train_acc, "avg_loss_per_ep": running_loss/len(trainloader)}
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"] or epoch == config["hparams"]["n_epochs"]:
            val_acc, avg_val_loss = evaluate(net, criterion, valloader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)

            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, save_path, net, None, log_file) # save best val ckpt
            

    ###########################
    # training complete
    ###########################
    n_gestures = config["hparams"]["model"]["num_classes"]
    log_dict = {f"acc_{n_gestures}_gestures": best_acc}
    log(log_dict, step, config)
