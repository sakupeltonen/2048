# Deep Q-Networks for 2048

This project implements DQN to the game of 2048. The agent uses Double DQN, afterstates, as well as weighted experience replay. 

## **Overview**

## Environment

### **Env2048**

2048 is a single-player sliding block puzzle game. The objective is to slide numbered tiles on a 4x4 grid to combine them and create a tile with the number 2048. The player slides tiles either up, down, left, or right. Two tiles with the same number can be merged into one, with double the value. The game is won when a tile with the value 2048 appears, though players can continue to play to achieve higher scores. The game is lost if there are no possible moves. 

### Wrappers
- **OnehotWrapper** Encodes the board into an array of one-hot vectors
- **PenalizeMovingUpWrapper** This wrapper penalizes the agent for moving up. It also has the capability to block the 'up' move by removing it from the available moves whenever other moves are available.
- **AfterstateWrapper** is an observation wrapper that returns the environment state after applying a given action, before spawning a random tile.
- **NextStateWrapper** This is an observation wrapper that adds information to help the agent understand the environment dynamics. For each possible move (left, right, down, up) **in the new state**, the information includes
  - reward (approximately normalized by dividing by `max_tile`)
  - whether the move is valid
  - whether the game ended (in one possible outcome for the move)

**RotationInvariantWrapper** The original 2048 game is symmetric w.r.t. rotation and reflection. This wrapper returns a rotation-invariant observation of the board. Actions taken by the agent are reversed corresponding to the rotation, so that their effect in the original environment is appropriate. The wrapper should help the agent to better understand the relevant effect of its actions. For example, consider the following state (*first picture*). The consequtive tiles are lined up nicely, which makes it easy to eventually combine them once the 64 is turned into another 256 tile.

<p align="left">
  <img src="/screenshots/0-2-0-0-0-0-0-0-0-0-0-0-64-256-512-1024.jpg" width="250" hspace="20" />
  <img src="/screenshots/64-2-512-1024-0-256-0-0-0-0-0-0-0-0-2-0.jpg" width="250" hspace="20" />
  <img src="/screenshots/0-2-0-0-0-0-0-0-0-256-0-0-64-2-512-1024.jpg" width="250" hspace="20" /> 
</p>

(*Second picture*): Moving up breaks the alignment, and now there is less space on the board to create new tiles. However, the game state looks quite different to the DQN, so it might be difficult to assess the effect of moving up. (*Third picure*): The state after moving up, when applying `RotationInvariantWrapper`. This state resembles the previous state, as the majority of tiles are in their original place, but the structure of the board is clearly different.  


## DQN Agent
<!-- The agent uses Double DQN and afterstates. The network takes in a (wrapped) observation of the environment. The output is a 4-dimensional vector corresponding to the Q-values for the 4 possible moves. The network architecture consists of 5 linear layers with ReLU activations for the first 4 layers (the last layer does not have an activation function since Q-values are not restricted to the unit interval). 
- **Experience replay** Implements an experience replay buffer with custom weights. A sum tree data structure allows sampling experiences from the buffer in $O(\log N)$ time, where N is the buffer size.
- **Frozenlake**: I tested the agent with the OpenAI Frozenlake environment. The current implementation is in another branch. --->

### Agent specifications
The specifications of an agent are stored in a JSON file: 

- `name`: Name of the agent. There should be a corresponding file in `/specs`. 
- `width, height`: Dimensions of the board.
- `prob_2`: Probability of spawning a 2-tile. $P(4) = 1-P(2)$
- `max_tile`: Maximum tile value in the environment. 2048 is the original goal of the game, but the player can also keep playing. The value affects the size of the model. 
- `epsilon_start`, `epsilon_final`, `epsilon_decay_last_episode`: The agent uses epsilon-greedy exploration. Epsilon decays linearly for the first `epsilon_decay_last_episode` episodes.
- `max_lr`, `min_lr`, `lr_cycle_length`: The CosineAnnealingLR scheduler from PyTorch is used to adjust the learning rate during training according to a cosine annealing schedule.
- `replay_size`: Number of experiences (single transition in the environment) stored in the experience replay buffer.
- `replay_start_size`: Minimum number of experiences acquired before starting training.
- `batch_size`: Batch size for learning.
- `gamma`: Discount factor for the MDP.
- `layer_size`: The size of the hidden layers in DQN.
- `sync_target_net_freq`: Frequency (in episodes) of syncing the target net and the training net in Double DQN.
- `test_freq`: Frequency (in episodes) of testing the net with greedy espiodes.
- `test_size`: Number of episodes per test.
- `human_played_games`: List of file names stored in `/games` as `.pkl` files, used as experiences.
- `recall_freq`: Frequency (in episodes) of re-inserting the human played games to the replay buffer. 

## Human Play
<!-- Play the game yourself. Board size and spawning probability can be customized. --->

