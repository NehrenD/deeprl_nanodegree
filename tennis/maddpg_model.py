import torch
import torch.nn as nn
import numpy as np
import ptan

#UTILS
class Config():

    #NETWORK

    ACTOR_FC1_UNITS = 128
    ACTOR_FC2_UNITS = 64
    CRITIC_FC1_UNITS = 128
    CRITIC_FC2_UNITS = 64
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-4
    TAU = 1e-4

    #REPLAY BUFFER
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 256

    GAMMA = 0.99

    device = 'cpu'
