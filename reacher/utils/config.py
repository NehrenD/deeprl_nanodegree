
class Config():

    #NETWORK

    ACTOR_FC1_UNITS = 64
    ACTOR_FC2_UNITS = 64
    CRITIC_FC1_UNITS = 64
    CRITIC_FC2_UNITS = 64
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    TAU = 1e-3
    WEIGHT_DECAY = 0  # L2 weight decay

    #REPLAY BUFFER
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 128

    GAMMA = 0.99

    #RUNS
    MAX_STEPS = 500
    TRAIN_STEPS = 20
    TRAIN_TIMES = 10

    device = 'cpu'
