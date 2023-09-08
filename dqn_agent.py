#!/usr/bin/env python3
from dqn_model import DQN
from environment import Env2048, OnehotWrapper, log2

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

"""
TODO
- test the agent every once in a while by running a couple of games with no exploration
- write proper printing or log to tensorboard
"""

from tensorboardX import SummaryWriter


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])



class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class DQNAgent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.score = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        # note : nothing prevents from taking moves that don't actually change the board. 
        # the agent should learn to not do this but make sure
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, info = self.env.step(action)
        self.score += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            _score = self.score
            self._reset()
            return _score
        return None  # score is not returned in the middle of an episode


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    # TODO this needs work
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument('--width', type=int, default=4,
                        help='Width of the board. Default is 4.')
    parser.add_argument('--height', type=int, default=4,
                        help='Height of the board. Default is 4.')
    parser.add_argument('--prob-2', type=float, default=0.9,
                        help='Probability of spawning a 2. Default is 0.9. P(4) = 1 - P(2).')
    parser.add_argument('--max-tile', type=int, default=512,
                        help='Maximum tile. Game is won once the maximum tile is reached.')
    parser.add_argument('--epsilon-start', type=float, default=1,
                        help='Initial value for epsilon-greedy policy. Decreases linearly in the number of episodes.')
    parser.add_argument('--epsilon-final', type=float, default=0.01,
                        help='Final value for epsilon-greedy policy. Decreases linearly in the number of episodes.')
    parser.add_argument('--epsilon-decay-last-frame', type=int, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--replay-size', type=int, default=20000)
    parser.add_argument('--replay-start-size', type=int, default=1000, 
                        help='Number of frames before to initiate replay buffer before starting training.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.95, help='Decay rate in Bellman equation')
    parser.add_argument('--sync-target-net-freq', type=int, default=5000, help='Frequency of syncing DQNs')
    
    
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = Env2048(width=args.width, height=args.height, prob_2=args.prob_2, max_tile=args.max_tile)
    env = OnehotWrapper(env)

    maxval = log2[args.max_tile] + 1
    net = DQN(maxval, args.height, args.width, 4).to(device)
    tgt_net = DQN(maxval, args.height, args.width, 4).to(device)
    
    writer = SummaryWriter(comment="-2048")
    print(net)

    buffer = ExperienceBuffer(args.replay_size)
    agent = DQNAgent(env, buffer)
    epsilon = args.epsilon_start
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    scores = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_score = None

    while True:
        frame_idx += 1
        epsilon = max(args.epsilon_final, args.epsilon_start -
                      frame_idx / args.epsilon_decay_last_frame)

        score = agent.play_step(net, epsilon, device=device)
        if score is not None:  # end of an episode
            scores.append(score)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_score = np.mean(scores[-100:])
            # print("%d: done %d games, score %.3f, "
            #       "eps %.2f, speed %.2f f/s" % (
            #     frame_idx, len(scores), m_score, epsilon,
            #     speed
            # ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("score_100", m_score, frame_idx)
            writer.add_scalar("score", score, frame_idx)
            if best_m_score is None or best_m_score < m_score:
                torch.save(net.state_dict(), "models/-best_%.0f.dat" % m_score)
                if best_m_score is not None:
                    print("Best score updated %.3f -> %.3f" % (
                        best_m_score, m_score))
                best_m_score = m_score
            

        if len(buffer) < args.replay_start_size:
            continue

        if frame_idx % args.sync_target_net_freq == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, args.gamma, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()