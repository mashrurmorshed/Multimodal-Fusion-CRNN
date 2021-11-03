from torch import optim


def get_optimizer(net, hparams):
    if hparams["optimizer"]["opt_type"] == "adadelta":
        optimizer = optim.Adadelta(net.parameters(), **hparams["optimizer"]["opt_kwargs"])
    elif hparams["optimizer"]["opt_type"] == "adamw":
        optimizer = optim.AdamW(net.parameters(), **hparams["optimizer"]["opt_kwargs"])
    else:
        raise ValueError(f"Unsupported optimizer {hparams['optimizer']['opt_type']}")
    return optimizer
