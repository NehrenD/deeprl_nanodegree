# Multi Agent Project Submission

## Project Overview

The project consists in training a DRL Agent to successfully solve the Tennis Unity environment

For the purpose of this project solving the environment is achieving an average score of +30 over 100 consecutive episodes.

## Tennis Unity Environment Details:

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

Solving the environment consists in achieving an average score of average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents.

### Getting Started

The current project has been implemented in a conda environment on Mac OSX

Unzip the file in the chosen directory and cd in the directory

To create an environment with the required dependencies:

```python
conda create -n tennis python=3.6
conda activate tennis
conda env update --file ./tennis_env.yml
conda activate tennis
```

### Zip Content
-README.md this file
-maddpg_model.py python file containing the config the agent and the actor and critic networks
-solved_state.ckp contains the checkpoint file with the trained parameters
-MultiAgent-Training.ipynb Notebook with the implementation of the training process
-Report.ipynb A notebook explaning the project and the solution with rewards plot and demonstration that agent solved the environment
-loss_actor_0.csv csv file from tensorboard with the loss of the actor of agent 1
-loss_actor_1.csv csv file from tensorboard with the loss of the actor of agent 2
-loss_critic_0.csv csv file from tensorboard with the loss of the critic of agent 1
-loss_critic_1.csv csv file from tensorboard with the loss of the critic of agent 2
-score_list.npy np file with scores
-tennis_eng.yml conda env config
-Tennis.app Unity environment

### Instructions 

#### For Training
Run MultiAgent-Training.ipynb
Execute notebook

#### For Evaluation
Open Report.ipynb and execute
