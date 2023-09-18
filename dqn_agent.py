#!/usr/bin/env python3
import numpy as np
import collections
import torch

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
        self.state = self.env.reset()[0]
        self.score = 0.0
        self.move_count = 0

    @torch.no_grad()
    def _evaluate(self, net, state):
        """get Q values and maximizing action for a given state"""
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        q_vals = net(state_v)
        _, act_v = torch.max(q_vals, dim=1)
        action = int(act_v.item())
        return q_vals, action

    @torch.no_grad()
    def play_step(self, net, max_moves, epsilon=0.0):
        # TODO get a list of valid moves. could also see if that is difficult to learn

        # note : nothing prevents from taking moves that don't actually change the board. 
        # the agent should learn to not do this but make sure
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self._evaluate(net, self.state)

        old_state = self.env.unwrapped.s
        # do step in the environment
        new_state, reward, is_done, _, info = self.env.step(action)

        # // For frozenlake, this is actually bad and prevents perfect strategies: can't move back safely
        # while new_state[old_state]:  # while hasn't moved. looks odd because we didn't wrap the old state to one-hot
        #     action = self.env.action_space.sample()
        #     new_state, reward, is_done, _, info = self.env.step(action)

        # TEMP: punish losing
        # if reward == 0 and is_done:
        #     reward = -1

        self.score += reward
        self.move_count += 1

        

        exp = Experience(self.state, action, reward,
                         is_done, new_state)

        # Compute priority. TODO clean up. TODO take is_done into account
        q_vals1, _ = self._evaluate(net, self.state)
        q_vals2, greedy_a2 = self._evaluate(net, new_state)
        # delta = reward + gamma * (1-is_done) * q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        delta = reward + (1-is_done) * q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        # TODO gamma should be a factor. make sure the is_done mask is correct
        priority = max(1, abs(delta))

        # priority = 1

        self.exp_buffer.append(exp, priority=priority) 
        
        # TEMP: end overly long / stuck episodes early
        if self.move_count > max_moves:
            is_done = True

        self.state = new_state
        if is_done:
            _score = self.score
            _move_count = self.move_count
            self._reset()
            return _score, _move_count
        return None  # score is not returned in the middle of an episode
