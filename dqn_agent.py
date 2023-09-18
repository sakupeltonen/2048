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
        """
        NOTE: In the frozenlake environment, it is quite hard to get stuck, even with a deterministic policy. 
        Even if the policy moves the agent towards the wall, there is always at least a 1/3 chance of moving away, even in the corner.

        The optimal policy may sometimes lead to the agent not moving for one step, since the randoness can turn the agent towards the wall. Fixing this with a random move would be detrimental! The agent relies on specs['max_moves'] to stop long episodes early. 
        """

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self._evaluate(net, self.state)

        # Take a step in the environment
        new_state, reward, is_done, _, info = self.env.step(action)

        self.score += reward
        self.move_count += 1

        exp = Experience(self.state, action, reward,
                         is_done, new_state)

        # Compute priority for experience
        # priority = 1

        q_vals1, _ = self._evaluate(net, self.state)
        q_vals2, greedy_a2 = self._evaluate(net, new_state)
        delta = reward + (1-is_done) * q_vals2[0][greedy_a2].item() - q_vals1[0][action].item()
        priority = max(1, abs(delta))

        self.exp_buffer.append(exp, priority=priority) 
        
        # End overly long / stuck episodes early
        if self.move_count > max_moves:
            is_done = True

        self.state = new_state
        if is_done:
            _score = self.score
            _move_count = self.move_count
            self._reset()
            return _score, _move_count
        return None  # score is not returned in the middle of an episode
