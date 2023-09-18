#!/usr/bin/env python3
import numpy as np
import collections
import torch
import random

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class DQNAgent:
    def __init__(self, env, exp_buffer, device='cpu'):
        self.env = env
        self.exp_buffer = exp_buffer
        self.device = device
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.score = 0.0

    @torch.no_grad()
    def _evaluate(self, net, state, available_moves):
        """get Q values and maximizing action for a given state"""
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        q_vals = net(state_v)
        
        valid_move_mask = torch.zeros(self.env.action_space.n).to(self.device)
        valid_move_mask[available_moves] = 1

        q_vals[0] *= valid_move_mask

        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())
        return q_vals, action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0):
        available_moves = self.env.unwrapped.available_moves()

        if np.random.random() < epsilon:
            action = random.choice(available_moves)
        else:
            _, action = self._evaluate(net, self.state, np.array(available_moves))

        # Take a step in the environment
        new_state, reward, is_done, info = self.env.step(action)

        self.score += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)

        # Compute priority for experience
        q_vals1, _ = self._evaluate(net, self.state, available_moves)
        next_available_moves = self.env.unwrapped.available_moves()  # TODO could think about saving these in the env
        q_vals2, greedy_a2 = self._evaluate(net, new_state, next_available_moves)
        delta = reward + (1-is_done) * q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        priority = max(1, abs(delta))


        self.exp_buffer.append(exp, priority=priority) 
        self.state = new_state
        if is_done:
            _score = self.score
            max_tile = self.env.unwrapped.board.max()
            move_count = self.env.unwrapped.legal_move_count
            self._reset()
            return (_score, max_tile, move_count)
        return None  # score is not returned in the middle of an episode