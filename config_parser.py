import yaml
import ast
import os
import pprint
import torch
import sys

def get_config(config_file):

    with open(config_file, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    if base_config["exp"]["wandb"]:
        if base_config["exp"]["wandb_api_key"] is not None:
            assert os.path.exists(base_config["exp"]["wandb_api_key"]), f"API key file not found at specified location {base_config['exp']['wandb_api_key']}."

    if base_config["exp"]["device"] == "auto":
        base_config["exp"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_config["hparams"]["device"] = base_config["exp"]["device"]
    return base_config


if __name__ == "__main__":
    config = get_config(sys.argv[1])
    pprint.pprint(config)