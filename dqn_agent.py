#!/usr/bin/env python3
import numpy as np
import collections
import torch
import random

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'board', 
                               'new_state', 'new_state_valid_moves', 'new_board'])
# state is a wrapped observation, e.g. with afterstates or onehot encoded
# board is the original board, stored for debugging purposes


class DQNAgent:
    def __init__(self, env, exp_buffer, device='cpu'):
        self.env = env
        self.exp_buffer = exp_buffer
        self.device = device
        self._reset()

    def _reset(self):
        self.state = self.env.reset() 
        self.board = self.env.unwrapped.board.copy()
        self.extra_obs = self.env.simulate_moves(self.board)  # store extra observation from NextStateWrapper for debugging
        self.score = 0.0

    @torch.no_grad()
    def _evaluate(self, net, state, valid_move_mask):
        """get Q values and maximizing action for a given state"""
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        q_vals = net(state_v)
        
        valid_move_mask = torch.tensor(valid_move_mask, dtype=torch.bool).to(self.device)
        q_vals[0][~valid_move_mask] = float('-inf') 

        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())
        return q_vals, action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0):
        available_move_mask = self.env.unwrapped.available_moves()

        if np.random.random() < epsilon:
            available_moves = [m for m in range(self.env.unwrapped.action_space.n) 
                               if available_move_mask[m]]
            action = random.choice(available_moves)
        else:
            _, action = self._evaluate(net, self.state, np.array(available_move_mask))

        # Take a step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        new_board = self.env.unwrapped.board.copy()

        self.score += reward

        # Compute priority for experience  # TODO should gamma appear here
        q_vals1, _ = self._evaluate(net, self.state, available_move_mask)
        next_available_moves_mask = self.env.unwrapped.available_moves()
        q_vals2, greedy_a2 = self._evaluate(net, new_state, next_available_moves_mask)
        if not is_done:
            delta = reward + q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        else:
            delta = reward - q_vals1[0][action].item() 
        priority = max(1, abs(delta))
        self.priority = priority   # TEMP save priority to log it in main

        # priority = 1

        exp = Experience(self.state, action, reward, is_done, self.board,
                    new_state, next_available_moves_mask, new_board)


        self.exp_buffer.append(exp, priority=priority) 
        self.state = new_state
        self.board = new_board
        self.extra_obs = self.env.simulate_moves(self.board)  # store extra observation from NextStateWrapper for debugging

        if is_done:
            _score = self.score
            max_tile = self.env.unwrapped.board.max()
            move_count = self.env.unwrapped.legal_move_count
            self._reset()
            return (_score, max_tile, move_count)
        return None  # score is not returned in the middle of an episode