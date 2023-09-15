#!/usr/bin/env python3
import numpy as np
import collections
import torch


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class DQNAgent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()  # TODO check
        self.score = 0.0
        self.invalid_move_count = 0

    @torch.no_grad()
    def _evaluate(self, net, state, device='cpu'):
        """get Q values and maximizing action for a given state"""
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(device)  # TODO is the device thing necessary
        q_vals = net(state_v)
        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())
        return q_vals, action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        # note : nothing prevents from taking moves that don't actually change the board. 
        # the agent should learn to not do this but make sure
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self._evaluate(net, self.state, device=device)

        # do step in the environment
        new_state, reward, is_done, info = self.env.step(action)

        # keep track of invalid moves for testing purposes
        # (random moves may well be invalid)
        if not info['valid_move']: 
            self.invalid_move_count += 1

        
        # we don't want the agent to get stuck with an invalid move 
        while not info['valid_move']:
            action = self.env.action_space.sample()
            new_state, reward, is_done, info = self.env.step(action)

        self.score += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)

        # Compute priority. TODO clean up
        q_vals1, _ = self._evaluate(net, self.state, device=device)
        q_vals2, greedy_a2 = self._evaluate(net, new_state, device=device)
        delta = reward + q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        priority = max(1, abs(delta))

        self.exp_buffer.append(exp, priority=priority) 
        self.state = new_state
        if is_done:
            # Also add experiences that should decrease the value of any encountered terminal state
            q_vals, _ = self._evaluate(net, self.state, device=device)
            
            for a in range(self.env.action_space.n):
                _exp = Experience(new_state, a, 0, True, new_state)

                priority = max(1, q_vals[0][a].item())
                self.exp_buffer.append(_exp, priority=priority)

            _score = self.score
            max_tile = self.env.unwrapped.board.max()
            move_count = self.env.unwrapped.legal_move_count
            invalid_count = self.invalid_move_count
            self._reset()
            return (_score, max_tile, move_count, invalid_count)
        return None  # score is not returned in the middle of an episode