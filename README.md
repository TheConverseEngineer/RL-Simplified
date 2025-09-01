# Overview
This repository stores a number of algorithms I coded while learning various Reinforcement Learning algorithms.

Everything included in this repository is for my own learning purposes and is in no way intended to be a 
production-ready implementation. The intention of this project
was very much to attempt to implement different RL algorithms from scratch (using only libraries like 
PyTorch and Gymnasium) and experiment with different methods.

### NOTE (More up-to-date version available)
This repository is a mirror of the repository hosted on the Georgia Tech Github Enterprise. Since these materials are actively in use for new-member training,
this repository has been temporarily reverted back to an outdated state. For the up-to-date version, login with your Georgia Tech SSO and access the repository at 
[github.gatech.edu/vparikh35](https://github.gatech.edu/vparikh35).

Not affiliated with Georgia Tech and interested in these resources? Send me an email at [vrajbparikh@gmail.com](mailto:vrajbparikh@gmail.com) and I'd love
to connect and send you the most up-to-date version!

# Navigation
Navigation directories is listed in reverse order of creation, 
with more recent/complex implementations listed first

### RLITE
This directory stores my attempt at building a general purpose RL library for PyTorch.
The library includes a variety of utility methods to generate various forms of training batches,
including a plethora of comments and debug statements.
In the samples folder, a number of custom implementations are included.
Most of these implementations were tested on CartPole, LunarLander, and Acrobot at the very least.
##### Included Implementations:
- PPO with clipped objective function
- Actor-Critic
- REINFORCE
- DQN
- Cross-Entropy method

### DQN
Includes a deep q-learning model that was successfully trained to play
Atari Pong with a target score of 18. A rainbow DQN implementation is
also provided

### Utils
My first attempt at creating a general-purpose RL library. This module is
used by many of the algorithms not found in the RLite folder.

### Policy-Based
Earlier iterations of my REINFORCE and actor-critic implementations (outdated by the RLite/samples implementation)

### Q-Learning
Includes a simple discrete q-learning agent that can solve FrozenLake

### Sutton-And-Barto
Includes solutions to the tic-tac-toe and bandit machine problems described
in the textbook by Sutton and Barto. The bandit-walks folder compares the performance
of the following agent types on both fixed and moving multi-arm slot machines.
- Averaging agent
- Stepping agent
- Optimistic agent
- Upper confidence bound (UCB) agent.


