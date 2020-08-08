import logging
import time
import sys
import torch
import numpy as np
import scipy as sp
import scipy.stats
 
def time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    
def create_logger(logger_name):

    logger = logging.getLogger(logger_name)

    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
   
    logfile_name = time_now().replace(" ", "_") + "_" + logger_name
    logfile_name = "./logs/" + logfile_name + ".log"
    LogFile = logging.FileHandler(logfile_name)
    LogFile.setFormatter(formatter)
    Console = logging.StreamHandler(stream=sys.stdout)
    Console.setFormatter(formatter)
    logger.addHandler(LogFile)
    logger.addHandler(Console)

    return logger, logfile_name

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h