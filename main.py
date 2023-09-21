from dqn_model import DQN
from dqn_agent import DQNAgent
from environment import Env2048, OnehotWrapper, AfterstateWrapper, RotationInvariantWrapper, log2
from experience_replay import ExperienceBuffer

import argparse
import time
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

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
        # Double Q-learning
        next_q_values = net(next_states_v)
        next_q_values[~next_valid_moves_v] = float('-inf')
        next_state_actions = next_q_values.max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        
        
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default="DQN-4x4")
    parser.add_argument("--cuda", default=False, 
                        action="store_true", help="Enable cuda")
    parser.add_argument("--save-model", default=False, action="store_true")
    parser.add_argument("--net-file", default=None)
    parser.add_argument("--session-data-file", default=None)
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
    env = RotationInvariantWrapper(env)
    env = AfterstateWrapper(env)
    env = OnehotWrapper(env)

    maxval = log2[specs['max_tile']] + 1

    net_args = (maxval, specs['height'], specs['width'], specs['layer_size'], 4)
    if args.net_file:
        path = os.path.join(script_dir, args.net_file)
        net = DQN.from_file(args.net_file, device, *net_args)
    else:
        net = DQN(*net_args).to(device)

    tgt_net = DQN(*net_args).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    writer = SummaryWriter(comment=f"-{args.agent_name}")
    print(net)

    buffer = ExperienceBuffer(specs['replay_size'])
    agent = DQNAgent(env, buffer, device=device)

    
    optimizer = optim.Adam(net.parameters(), lr=specs['max_lr'])
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=specs['lr_cycle_length']/2, eta_min=specs['min_lr'])

    if not args.session_data_file:
        step_idx = 0
        episode_idx = 0
    else:
        path = os.path.join(script_dir, args.session_data_file)
        session_data = json.load(open(path, "r"))
        step_idx = session_data['step_idx']
        episode_idx = session_data['episode_idx']

    episode_start = time.time()
    best_test_score = None

    while True:
        # Update epsilon
        epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                    episode_idx / specs['epsilon_decay_last_episode'])
        
        # epsilon = 0

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

            # Testing
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
                    now = datetime.now()
                    filename = now.strftime('%d%b-%H-%M') + args.agent_name

                    torch.save(net.state_dict(), f"models/{filename}.dat")

                    # Save related session data to be able to continue training from this point
                    session_data = {'episode_idx': episode_idx, 'step_idx': step_idx}
                    with open('session_data/' + filename + '.json', 'w') as f:
                        json.dump(session_data, f)

                    best_test_score = m_test_score


            episode_idx += 1

            # Update target net parameters  # TODO where should this be placed? What is a good value for tau compared to the learning rate
            # for target_param, param in zip(tgt_net.parameters(), net.parameters()):
            #     target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + \
            #                                 param.data * specs['tau'])

            # TEMP
            if episode_idx % specs['sync_target_net_freq'] == 0:
                tgt_net.load_state_dict(net.state_dict())


        if len(buffer) < specs['replay_start_size']:
            continue
        

        optimizer.zero_grad()
        batch = buffer.sample(specs['batch_size'])
        loss_t = calc_loss(batch, net, tgt_net, specs['gamma'], device=device)
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t.detach().item(), step_idx)
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], step_idx)

        lr_scheduler.step()

        step_idx += 1