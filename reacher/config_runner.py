from unityagents import UnityEnvironment
import numpy as np
import sys
from collections import deque
from agents.ddpg_agent import DDPGAgent
from utils.config import Config
import logging

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "reacher.log"

def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(FORMATTER)
   return console_handler

def get_file_handler():
   file_handler = logging.FileHandler(LOG_FILE)
   file_handler.setFormatter(FORMATTER)
   return file_handler

def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG) # better to have too much log than not enough
   logger.addHandler(get_console_handler())
   logger.addHandler(get_file_handler())
   # with this pattern, it's rarely necessary to propagate the error up to parent
   logger.propagate = False
   return logger


def main():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("filename")
    logger = get_logger("Main")
    run_logger = get_logger("RUN")
    config_grid = {

        #NETWORK
        'FC1_UNITS' : [32,64,128],
        'FC2_UNITS': [32, 64, 128],
        'ACTOR_LR': [1e-2, 1e-3,1e-4],
        'CRITIC_LR': [1e-2, 1e-3, 1e-4],
        'TAU': [1e-2, 1e-3, 1e-4],
        'NOISE_SIGMA': [0.1, 0.3, 0.5],

        #REPLAY BUFF
        'BUFFER_SIZE': [1e4,1e5,1e6],
        'BATCH_SIZE': [64, 128, 256],

        'MAX_STEPS': [ 200,300,400],
        'TRAIN_STEP': [10,50,100],
        'TRAIN_TIME': [10, 50, 100],

    }

    n_samples = 200
    ckeys = list(config_grid.keys())
    configs = []
    experiments = np.array([np.random.choice([0, 1, 2], n_samples*2) for _ in range(len(ckeys))])
    experiments = np.unique(experiments, axis=1)[:, :n_samples].T

    for i in range(n_samples):
        e = experiments[i]
        c = {k: config_grid[k][e[i]] for i, k in enumerate(ckeys)}
        configs.append(c)

    logger.info(f"Testing {len(configs)} Configurations")

    best_config = -1
    max_score = -1

    env = UnityEnvironment(file_name='Reacher-2.app', no_graphics=True)

    for i, c in enumerate(configs):
        config = Config()
        config.CRITIC_FC1_UNITS = c['FC1_UNITS']
        config.ACTOR_FC1_UNITS = c['FC1_UNITS']
        config.CRITIC_FC2_UNITS = c['FC2_UNITS']
        config.ACTOR_FC2_UNITS = c['FC2_UNITS']
        config.LR_CRITIC = c['CRITIC_LR']
        config.LR_ACTOR = c['ACTOR_LR']
        config.TAU = c['TAU']
        config.NOISE_SIGMA = c['NOISE_SIGMA']

        config.BUFFER_SIZE = int(c['BUFFER_SIZE'])
        config.BATCH_SIZE = int(c['BATCH_SIZE'])
        config.MAX_STEPS = c['MAX_STEPS']
        config.TRAIN_STEPS = c['TRAIN_STEP']
        config.TRAIN_STEPS = c['TRAIN_TIME']


        logger.info(f"NETWORK1: actor_fc1={config.ACTOR_FC1_UNITS}, actor_fc2={config.ACTOR_FC2_UNITS}, critic_fc1={config.CRITIC_FC1_UNITS}, critic_fc1={config.CRITIC_FC2_UNITS}")
        logger.info(f"NETWORK2: actor_LR={config.LR_ACTOR}, critic_LR={config.LR_CRITIC}, tau={config.TAU}, noise_theta={config.NOISE_THETA}, noise_sigma={config.NOISE_SIGMA}")
        logger.info(f"REPLAY_BUFF: rb_size={config.BUFFER_SIZE}, batch_size={config.BATCH_SIZE}")
        logger.info(f"RUN CONFIG: max_steps={config.MAX_STEPS}, train_stpes={config.TRAIN_STEPS}, train_times={config.TRAIN_TIMES}")

        run_max_score = run(env,config,run_logger)
        logger.info(f"max_score: {run_max_score}")

        if run_max_score > max_score:
            max_score = run_max_score
            best_config = i

    logger.info(f"Best score: {max_score, }config {configs[best_config]}")


def run(env, config,logger):



    seed = 1234
    n_epochs = 5000
    print_every = 10
    avg_score_target = 10


    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    num_agents = 20
    env_info = env.reset(train_mode=True)[brain_name]

    agent = DDPGAgent(state_size, action_size, seed,config)

    scores_deque_10 = deque(maxlen=10)
    scores_deque_50 = deque(maxlen=50)
    scores_deque_100 = deque(maxlen=100)
    scores = []
    cons_zeros = 0

    max_best_avg = -1

    for epoch in range(1, n_epochs + 1):

        states = env_info.vector_observations
        agent.reset()

        total_scores = np.zeros(num_agents)

        for t in range(config.MAX_STEPS):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            total_scores += rewards

            _ = [agent.add_to_memory(states[i], actions[i], rewards[i], next_states[i], dones[i]) for i in
                 range(num_agents)]

            if t % config.TRAIN_STEPS == 0:
                for _ in range(config.TRAIN_TIMES):
                    agent.learning_step()

            states = next_states

        scores_deque_10.extend(total_scores)
        scores_deque_50.extend(total_scores)
        scores_deque_100.extend(total_scores)
        scores.extend(total_scores)

        if epoch % print_every == 0:

            avg_scores_10 = np.asanyarray(scores_deque_10).mean()
            avg_scores_50 = np.asanyarray(scores_deque_50).mean()
            avg_scores_100 = np.asanyarray(scores_deque_100).mean()

            if avg_scores_100 > max_best_avg:
                max_best_avg = avg_scores_100

            if avg_scores_10 < 0.0001:
                cons_zeros +=1

            logger.info(f"Epoch: {epoch} \tavg_score_10: {avg_scores_10}\tavg_score_50: {avg_scores_50}\tavg_score_100: {avg_scores_100}")

            if avg_scores_100 > avg_score_target:
                logger.info("Enviroment Solved!")
                success = True
                break

            if (epoch >= 500 and avg_scores_100 < 1) or cons_zeros >= 5:
                logger.info("Environment Failed")
                break

    return max_best_avg

if __name__ == '__main__':
    main()