# Temporal Difference Learning for 2048

This project implements TD($\lambda$) to the game of 2048.

## **Overview**

2048 is a sliding tile puzzle game where the objective is to combine tiles with the same number to create larger numbers, with the ultimate goal of achieving the 2048 tile. 

## **Features**

- **Environment**: 
- **TD Agent**: 
- **Human play & visualization**: Play by yourself or replay games played by an agent. 

## **Agent specs**
The specifications of an agent are stored in a JSON file: 

- `name`: Name of the agent.
  - When initializing an agent with a given name, the trained agent is automatically saved to `agents/{name}.pkl`
  - An agent with a given name can be loaded using `TDAgent.load(specs)`
- `width, height`: Dimensions of the board.
- `prob_2`: Probability of spawning a 2-tile. $P(4) = 1-P(2)$
- `max_tile`: The game is won when a tile with value `max_tile` is obtained. 2048 is the original goal of the game. A smaller `max_tile` can be used to speed up training when testing different algorithms. A larger value of  `max_tile` increases the size of the look-up tables.
- `meta_learning_rate`: Constant learning rate used on top of the adaptive Temporal Coherence algorithm.
- `trace_decay`: $\lambda$
- `cut_off_weight`: Specifies a limited update horizon for the implementation of TD($\lambda$). Updates for previous states $s_{t-k}$ with a weight $\lambda^{k}$ less than `cut_off_weight` are omitted.
- `optimistic_init`: Initial value for the look-up tables. Larger values encourage exploration.
- `layout` is a list of 2-dimensional lists. Each entry is shaped like the board, specifying a layout for an n-tuple. $n$ corresponds to the (maximum) number of 1s in a layout. 
- `symmetries` is a boolean variable. When false, the n-tuple network is simply given by `layout`. When true, 7 copies of each n-tuple are added, corresponding to the rotational and reflectional symmetries. The copies use the same look-up table as the original n-tuple. `symmetries` should be set to false when the board is non-rectangular
- `update_freq` and `save_freq` specify the frequency (in episodes) of printing and saving agent information. 
