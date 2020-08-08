import os
import torch.nn as nn
import torch.optim
from tqdm import tqdm, trange
import time
from colorama import Fore, Back, Style
import logging
import argparse
import yaml
from easydict import EasyDict
from models import PARN
from dataloader import get_dataloader
import utils
   
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--gpu", type=int, default=0)
    parser.add_argument("-w","--num_workers", type=int, default=3)
    args = parser.parse_args()

    config.gpu = args.gpu
    config.num_workers = args.num_workers

    model = PARN(config)
    logger_name = config.metric_type
    logger, logfile_name = utils.create_logger(logger_name)
    model.set_logger(logger, logfile_name)
    model.logger.info(config)

    model.train_process()
    model.test_process()

if __name__ == "__main__":
    main()
