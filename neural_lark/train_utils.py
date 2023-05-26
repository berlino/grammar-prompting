import os
import sys
import math
import random
import time
import logging
import argparse
import torch
import numpy as np

import logging

def setup_logger():
    logger = logging.getLogger("neural_lark")
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()


def setup_logger_file(logger, log_dir):
    """
    Send info to console, and detailed debug information in logfile
    (Seems not working)
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logfile_path = os.path.join(log_dir, f"log_{timestr}.txt")
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    # sh.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile_path)
    fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(sh)  # log to console as well

    logger.info("Logging to {}".format(logfile_path))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

