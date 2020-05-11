# Navigation Project Submission

## Project Overview

The project consists in training a DRL Agent to successfully solve the Banana Unity environment

For the purpose of this project solving is achieving an average score of at least 13 points over 100 episodes.

## Banana Unity Environment Details:

The environment consists in a square world with randomly placed yellow and blue bananas.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
The action space is  discrete one dimensional with action 'Forward', 'Backward', 'Left', 'Right'
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

Goal of the agent is to collect as many yellow bananas as possible and avoid blue bananas.

### Getting Started

The current project has been implemented in a conda environment on Mac OSX

Unzip the file in the chosen directory and cd in the directory

To create an environment with the required dependencies:

```python
conda create -n navigation python=3.6
conda activate navigation
conda env update --file ./navigation_env.yml
conda activate navigation
```

### Zip Content
-README.md this file
-agent  package containing double_ddqn.pyt implementation of the Agent
-network package containing duelin_dqn.py implementing the Network
-util package containing some utilities
-checkpoint_solution.pth contains the checkpoint file with the trained parameters
-Navigation_Project_Training.ipynb Notebook with the implementation of the training process
-Report.ipynb A notebook explaning the project and the solution with rewards plot and demonstration that agent solved the environment
-rewards_plot.eps image for the rewards plot
-navigation_eng.yml conda env config
-Banana.app Unity environment

### Instructions 

#### For Training
Run Jupyter Notebook and open Navigation_Project_Training.ipynb
Execute notebook

#### For Evaluation
Open Report.ipynb and execute
