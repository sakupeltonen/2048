# Deep Q-Networks for 2048

This project implements DQN to the game of 2048. The agent uses Double DQN, afterstates, as well as weighted experience replay. 

## **Overview**

2048 is a single-player sliding block puzzle game. The objective is to slide numbered tiles on a 4x4 grid to combine them and create a tile with the number 2048. The player slides tiles either up, down, left, or right. Two tiles with the same number can be merged into one, with double the value. The game is won when a tile with the value 2048 appears, though players can continue to play to achieve higher scores. The game is lost if there are no possible moves. 

## **Features**

- **Environment**: OpenAI Gym environment for 2048.
  - `OnehotWrapper`: Converts discrete action spaces in gym environments into one-hot encoded representations
  - `AfterstateWrapper` is a wrapper that modifies the environment to return the state after applying a given action but before spawning a random tile.
- **DQN Agent**: The agent uses Double DQN and afterstates.
  - The network takes in an (afterstate) observation of the board encoded as a _width_ x _height_ x log(_maxtile_) sized one-hot vector. The output is a 4-dimensional vector corresponding to the Q-values for the 4 possible moves. The network architecture consists of 3 linear layers with ReLU activations for the first 2 layers (the last layer does not have an activation function since Q-values are not restricted to the unit interval). 
- **Experience replay** Implements an experience replay buffer with custom weights. A sum tree data structure allows sampling experiences from the buffer in $O(\log N)$ time, where N is the buffer size.
- **Human play**: Play the game yourself. Board size and spawning probability can be customized.
- **Frozenlake**: I tested the agent with the OpenAI Frozenlake environment. The current implementation is in another branch. 

## **Agent specifications**
The specifications of an agent are stored in a JSON file: 

- `name`: Name of the agent.
- `width, height`: Dimensions of the board.
- `prob_2`: Probability of spawning a 2-tile. $P(4) = 1-P(2)$
- `max_tile`: The game is won when a tile with value `max_tile` is obtained. 2048 is the original goal of the game. A smaller `max_tile` can be used to speed up training when testing different algorithms.
- `epsilon_start`, `epsilon_final`, `epsilon_decay_last_episode`: The agent uses epsilon-greedy exploration. Epsilon decays linearly for the first `epsilon_decay_last_episode` episodes.
- `gamma`: Discount factor for the MDP.
- `tau`: Soft update parameter for double DQN.

