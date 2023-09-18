from dqn_model import DQN
from dqn_agent import DQNAgent
from environment import Env2048, OnehotWrapper, log2
from experience_replay import ExperienceBuffer

import argparse
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states, next_valid_moves_masks = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_valid_moves_v = torch.tensor(next_valid_moves_masks, dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # Q-learning
        # next_q_values = net(next_states_v)
        # next_q_values[~next_valid_moves_v] = float('-inf')
        # next_state_values = next_q_values.max(1)[0]

        # Double Q-learning # TODO check
        next_q_values = net(next_states_v)
        next_q_values[~next_valid_moves_v] = float('-inf')
        next_state_actions = next_q_values.max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        
        
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default="DQN-4x4")
    parser.add_argument("--cuda", default=False, 
                        action="store_true", help="Enable cuda")
    parser.add_argument("--save-model", default=False, action="store_true")
    args = parser.parse_args()

    specs_file = f"specs/{args.agent_name}.json"

    device = torch.device("cuda" if args.cuda else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, specs_file)
    specs = json.load(open(path, "r"))

    env = Env2048(width=specs['width'], 
                  height=specs['height'], 
                  prob_2=specs['prob_2'], 
                  max_tile=specs['max_tile'])
    env = OnehotWrapper(env)

    maxval = log2[specs['max_tile']] + 1
    net = DQN(maxval, specs['height'], 
              specs['width'], specs['layer_size'], 4).to(device)
    tgt_net = DQN(maxval, specs['height'], 
                  specs['width'], specs['layer_size'], 4).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    writer = SummaryWriter(comment=f"-{args.agent_name}")
    print(net)

    buffer = ExperienceBuffer(specs['replay_size'])
    agent = DQNAgent(env, buffer, device=device)

    epsilon = specs['epsilon_start']
    optimizer = optim.Adam(net.parameters(), lr=specs['learning_rate'])

    step_idx = 0
    episode_idx = 0
    episode_start = time.time()
    best_test_score = None

    while True:
        res = agent.play_step(net, epsilon=epsilon)

        # End of an episode
        if res is not None:  
            score, max_tile, move_count = res

            # Time episode
            episode_duration = time.time() - episode_start
            writer.add_scalar("episode duration", episode_duration, episode_idx)
            episode_start = time.time()
            # TODO this doesn't work during tests.
            # In general should restructure, so that the game loop is the same for training and testing

            # Log average of statistics
            writer.add_scalar("score", score, episode_idx)
            writer.add_scalar("max tile", max_tile, episode_idx)
            writer.add_scalar("epsilon", epsilon, episode_idx)
            writer.add_scalar("move count", move_count, episode_idx)

            # Update epsilon
            epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                      episode_idx / specs['epsilon_decay_last_episode'])
            
            # Testing  # TODO seems to get stuck here, but not on the first test. 
            if episode_idx % specs['test_freq'] == 0:
                test_scores = []
                test_max_tiles = []
                for i in range(specs['test_size']):
                    # Play a game with epsilon=0
                    while True:
                        res = agent.play_step(net)
                        if res:
                            score, max_tile, move_count = res
                            test_scores.append(score)
                            test_max_tiles.append(max_tile)

                            break
                
                # Print and log test statistics
                m_test_score = round(np.mean(test_scores))
                m_max_tile = round(np.mean(test_max_tiles))
                writer.add_scalar("mean greedy score", m_test_score, episode_idx)
                writer.add_scalar("mean greedy max tile", m_max_tile, episode_idx)

                print(f'Episode {episode_idx}: average score {m_test_score}, average max tile {m_max_tile}')
            
                # Save improved model
                if args.save_model and (best_test_score is None or best_test_score < m_test_score):
                    torch.save(net.state_dict(), "models/-best_%.0f.dat" % m_test_score)
                    if best_test_score is not None:
                        print("Best score updated %.3f -> %.3f" % (
                            best_test_score, m_test_score))
                    best_test_score = m_test_score

            episode_idx += 1

            # Update target net parameters  # TODO where should this be placed? What is a good value for tau compared to the learning rate
            # for target_param, param in zip(tgt_net.parameters(), net.parameters()):
            #     target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + \
            #                                 param.data * specs['tau'])

            # TEMP
            # if episode_idx % specs['sync_target_net_freq'] == 0:
            #     tgt_net.load_state_dict(net.state_dict())
            

        if len(buffer) < specs['replay_start_size']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(specs['batch_size'])
        loss_t = calc_loss(batch, net, tgt_net, specs['gamma'], device=device)
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t.detach().item(), step_idx)

        step_idx += 1