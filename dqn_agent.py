#!/usr/bin/env python3
from dqn_model import DQN
from environment import Env2048, OnehotWrapper, log2

import argparse
import time
import numpy as np
import collections
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

"""
- Syncing the networks seems to cause a spike in the loss. Might look up ways to reduce this
- Try with no or less exploration? Seems to actually converge when it's not learning based on random moves

I wonder what the agent thinks about states that are actually lost, in the sense that can't move anywhere. They are never updated, because they don't get into experiences

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
        self.invalid_move_count = 0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, testing=False, device="cpu"):
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

        # keep track of invalid moves for testing purposes
        # (random moves may well be invalid)
        if not info['valid_move']: 
            self.invalid_move_count += 1

        
        # we don't want the agent to get stuck with an invalid move 
        while not info['valid_move']:
            action = env.action_space.sample()
            new_state, reward, is_done, info = self.env.step(action)

        self.score += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            # Also add experiences that should decrease the value of any encountered terminal state
            for a in range(env.action_space.n):
                _exp = Experience(new_state, a, 0, True, new_state)
                self.exp_buffer.append(_exp)

            _score = self.score
            max_tile = env.unwrapped.board.max()
            move_count = env.unwrapped.legal_move_count
            invalid_count = self.invalid_move_count
            self._reset()
            return (_score, max_tile, move_count, invalid_count)
        return None  # score is not returned in the middle of an episode


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    # TODO: track average difference to test if converges at 

    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    # TODO some spikes in the loss. make sure that the done mask is working

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + \
                                   rewards_v
    loss = nn.MSELoss()(state_action_values,
                        expected_state_action_values)
    frac_loss = (loss / expected_state_action_values.sum()).detach().item()
    return loss, frac_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--specs-file", default="specs/DQN-4x4.json")
    parser.add_argument("--specs-file", default="specs/DQN-2x3.json")
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    
    args = parser.parse_args()

    # TEMP
    agent_name = '2x3'
    args.specs_file = f"specs/DQN-{agent_name}.json"

    device = torch.device("cuda" if args.cuda else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, args.specs_file)
    specs = json.load(open(path, "r"))

    env = Env2048(width=specs['width'], height=specs['height'], prob_2=specs['prob_2'], max_tile=specs['max_tile'])
    env = OnehotWrapper(env)

    maxval = log2[specs['max_tile']] + 1
    net = DQN(maxval, specs['height'], specs['width'], specs['layer_size'], 4).to(device)
    tgt_net = DQN(maxval, specs['height'], specs['width'], specs['layer_size'], 4).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    writer = SummaryWriter(comment=f"-{agent_name}")
    print(net)

    buffer = ExperienceBuffer(specs['replay_size'])
    agent = DQNAgent(env, buffer)
    epsilon = specs['epsilon_start']
    optimizer = optim.Adam(net.parameters(), lr=specs['learning_rate'])

    scores = []
    max_tiles = []
    episode_durations = []  # time in seconds
    move_counts = []  # steps
    frac_losses = []
    losses = []
    stats_period = 500  # 

    episode_idx = 0
    episode_start = time.time()
    best_m_score = None

    while True:
        res = agent.play_step(net, epsilon, device=device)  # also get max tile (no legality of move since there are random moves)

        if res is not None:  # end of an episode
            score, max_tile, move_count, _ = res
            scores.append(score)
            max_tiles.append(max_tile)
            move_counts.append(move_count)

            episode_duration = time.time() - episode_start
            episode_start = time.time()
            episode_durations.append(episode_duration)

            if episode_idx >= stats_period:
                writer.add_scalar("mean score", np.mean(scores[-stats_period:]), episode_idx)
                writer.add_scalar("mean max tile", np.mean(max_tiles[-stats_period:]), episode_idx)
                writer.add_scalar("epsilon", epsilon, episode_idx)
                writer.add_scalar("mean episode duration", np.mean(episode_durations[-stats_period:]), episode_idx)
                writer.add_scalar("mean move count", np.mean(move_counts[-stats_period:]), episode_idx)

                writer.add_scalar("average fractional loss", np.mean(frac_losses[-stats_period:]), episode_idx)
                writer.add_scalar("average loss", np.mean(losses[-stats_period:]), episode_idx)

            epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                      episode_idx / specs['epsilon_decay_last_episode'])
            
            if episode_idx % specs['test_freq'] == 0:
                test_scores = []
                test_max_tiles = []
                invalid_counts = []
                for i in range(specs['test_size']):
                    while True:
                        res = agent.play_step(net, testing=True, device=device)  # epsilon=0
                        # also get max tile and legality of move 
                        if res:
                            score, max_tile, move_count, invalid_move_count = res
                            test_scores.append(score)
                            test_max_tiles.append(max_tile)
                            invalid_counts.append(invalid_move_count)
                            break
                
                m_test_score = round(np.mean(test_scores))
                m_max_tile = round(np.mean(test_max_tiles))
                m_invalid_count = round(np.mean(invalid_counts))
                print(f'Episode {episode_idx}: average score {m_test_score}, average max tile {m_max_tile}, average #invalid {m_invalid_count}')
                writer.add_scalar("greedy Test Score", m_test_score, episode_idx // specs['test_freq'])
                writer.add_scalar("greedy Max tile", m_max_tile, episode_idx // specs['test_freq'])
                writer.add_scalar("greedy Average number of invalid", m_invalid_count, episode_idx // specs['test_freq'])
            
                # if best_m_score is None or best_m_score < m_test_score:
                #     torch.save(net.state_dict(), "models/-best_%.0f.dat" % m_test_score)
                #     if best_m_score is not None:
                #         print("Best score updated %.3f -> %.3f" % (
                #             best_m_score, m_test_score))
                #     best_m_score = m_test_score

            episode_idx += 1



            # if episode_idx % specs['sync_target_net_freq'] == 0:
            #     tgt_net.load_state_dict(net.state_dict())

            
            for target_param, param in zip(tgt_net.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + param.data * specs['tau'])
            

        if len(buffer) < specs['replay_start_size']:
            continue
        

        optimizer.zero_grad()
        batch = buffer.sample(specs['batch_size'])
        loss_t, frac_loss = calc_loss(batch, net, tgt_net, specs['gamma'], device=device)
        loss_t.backward()
        losses.append(loss_t.detach())
        frac_losses.append(frac_loss)
        optimizer.step()

    # writer.close()






    # parser.add_argument('--width', type=int, default=4,
    #                     help='Width of the board. Default is 4.')
    # parser.add_argument('--height', type=int, default=4,
    #                     help='Height of the board. Default is 4.')
    # parser.add_argument('--prob-2', type=float, default=0.9,
    #                     help='Probability of spawning a 2. Default is 0.9. P(4) = 1 - P(2).')
    # parser.add_argument('--max-tile', type=int, default=512,
    #                     help='Maximum tile. Game is won once the maximum tile is reached.')
    # parser.add_argument('--epsilon-start', type=float, default=1,
    #                     help='Initial value for epsilon-greedy policy. Decreases linearly in the number of episodes.')
    # parser.add_argument('--epsilon-final', type=float, default=0.01,
    #                     help='Final value for epsilon-greedy policy. Decreases linearly in the number of episodes.')
    # parser.add_argument('--epsilon-decay-last-episode', type=int, default=10000)
    # parser.add_argument('--learning-rate', type=float, default=0.005)
    # parser.add_argument('--replay-size', type=int, default=20000)
    # parser.add_argument('--replay-start-size', type=int, default=1000, 
    #                     help='Number of frames before to initiate replay buffer before starting training.')
    # parser.add_argument('--batch-size', type=int, default=16)
    # parser.add_argument('--gamma', type=float, default=0.95, help='Decay rate in Bellman equation')
    # parser.add_argument('--sync-target-net-freq', type=int, default=200, help='Frequency of syncing DQNs')
    # parser.add_argument('--test-freq', type=int, default=200, help='Frequency of testing the agent')
    # parser.add_argument('--test-size', type=int, default=30, help='Number of games used to test the agent')