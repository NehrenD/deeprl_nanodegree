# Reacher Project Submission

## Project Overview

The project consists in training a DRL Agent to successfully solve the Reacher Unity environment

For the purpose of this project solving the environment is achieving an average score of +30 over 100 consecutive episodes.

## Reacher Unity Environment Details:

In this environment, a double-jointed arm can move to target locations. 
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
Each action is a vector with four numbers, corresponding to torque applicable to two joints with values between -1 and 1
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

The goal of the agent is to maintain its position at the target location for as many time steps as possible.


### Getting Started

The current project has been implemented in a conda environment on Mac OSX

Unzip the file in the chosen directory and cd in the directory

To create an environment with the required dependencies:

```python
conda create -n reacher python=3.6
conda activate reacher
conda env update --file ./reacher_env.yml
conda activate reacher
```

### Zip Content
-README.md this file
-util package containing some utilities
-ddpg_model_ptan.py python file containing the config the agent and the actor and critic networks
-learned_state.ckp contains the checkpoint file with the trained parameters
-Continuous_Control_Training.ipynb Notebook with the implementation of the training process
-Report.ipynb A notebook explaning the project and the solution with rewards plot and demonstration that agent solved the environment
-loss_actor.csv csv file from tensorboard with the loss of the actor
-loss_critic.csv csv file from tensorboard with the loss of the critic
-reward.csv csv file from tensorboard with the raw rewards
-reward_100.csv csv file from tensorboard with the rewards of the trailing 100 episodes
-navigation_eng.yml conda env config
-Reacher.app Unity environment

### Instructions 

#### For Training
Run Continuous_Control_Training.ipynb.ipynb
Execute notebook

#### For Evaluation
Open Report.ipynb and execute
