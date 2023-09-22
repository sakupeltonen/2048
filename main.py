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
import shutil

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


def get_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default="DQN-4x4")
    parser.add_argument("--cuda", default=False, 
                        action="store_true", help="Enable cuda")
    parser.add_argument("--save-model", default=False, action="store_true")
    parser.add_argument("--net-file", default=None, help="Location of DQN model")
    parser.add_argument("--session-data-file", default=None, help="Location of session data file for continuing a previous training session")
    parser.add_argument("--colab", default=False, action="store_true", help="Set to True when running in Google Colab to make data persistent")
    args = parser.parse_args()
    return args


def test_greedy(agent, test_size, net, writer):
    test_scores = []
    test_max_tiles = []
    for _ in range(test_size):
        # Play a game with epsilon=0
        while True:
            res = agent.play_step(net)
            if res:
                score, max_tile, _ = res
                test_scores.append(score)
                test_max_tiles.append(max_tile)

                break
    
    # Print and log test statistics
    m_test_score = round(np.mean(test_scores))
    m_max_tile = round(np.mean(test_max_tiles))
    writer.add_scalar("mean greedy score", m_test_score, episode_idx)
    writer.add_scalar("mean greedy max tile", m_max_tile, episode_idx)

    print(f'Episode {episode_idx}: average score {m_test_score}, average max tile {m_max_tile}')



def save_model(net, session_data, colab=False, drive_dir=None, model_dir="models", 
               session_data_dir="session_data"):
    """Save DQN net and session data related to training. """
    now = datetime.now()
    timestamp = now.strftime('%d%b-%H-%M')
    model_filename = os.path.join(model_dir, f'{timestamp}{args.agent_name}.dat')
    session_data_filename = os.path.join(session_data_dir, f'{timestamp}{args.agent_name}.json')

    torch.save(net.state_dict(), model_filename)
    
    with open(session_data_filename, 'w') as f:
        json.dump(session_data, f)

    if colab: 
        shutil.copyfile(model_filename, 
                        os.path.join(drive_dir, os.path.basename(model_filename)))
        shutil.copyfile(session_data_filename, 
                        os.path.join(drive_dir, os.path.basename(session_data_filename)))


def copy_logs(base_src_dir, log_name, base_dst_dir):
    """
       Logs are saved in the runs directory, where each run is created its own folder, base_src_dir/log_name
       e.g. /runs/Sep20_00-58-36_sakus-mbp-2.lan-DQN-1x4/events.out.tfevents.1695160716.sakus-mbp-2.lan
        The function copies the contents to Google Drive in base_dst_dir/log_name    
    """
    src_dir = os.path.join(base_src_dir, log_name)
    dst_dir = os.path.join(base_dst_dir, log_name)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        shutil.copy2(s, d)  # copy2 preserves file metadata



if __name__ == "__main__":
    args = get_program_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Agent specifications
    specs_file = f"specs/{args.agent_name}.json"
    path = os.path.join(script_dir, specs_file)
    specs = json.load(open(path, "r"))

    # Create and wrap environment
    env = Env2048(width=specs['width'], 
                  height=specs['height'], 
                  prob_2=specs['prob_2'], 
                  max_tile=specs['max_tile'])
    env = RotationInvariantWrapper(env)
    env = AfterstateWrapper(env)
    env = OnehotWrapper(env)

    maxval = log2[specs['max_tile']] + 1

    drive_dir = "/content/drive/MyDrive/2048"

    device = torch.device("cuda" if args.cuda else "cpu")

    # Create Deep Q networks
    net_args = (maxval, specs['height'], specs['width'], specs['layer_size'], 4)
    if args.net_file:
        path = os.path.join(script_dir, args.net_file)
        net = DQN.from_file(args.net_file, device, *net_args)
    else:
        net = DQN(*net_args).to(device)

    tgt_net = DQN(*net_args).to(device)
    tgt_net.load_state_dict(net.state_dict())
    print(net)
    
    # Initialize logger
    now = datetime.now()  # current date and time
    timestamp = now.strftime('%d%b-%H-%M')
    log_dir = f"{timestamp}-{args.agent_name}"
    writer = SummaryWriter(log_dir='runs/'+log_dir)
    
    # Initialize experience replay buffer 
    buffer = ExperienceBuffer(specs['replay_size'])

    # Initialize agent
    agent = DQNAgent(env, buffer, device=device)

    # Initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=specs['max_lr'])
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=specs['lr_cycle_length']/2, eta_min=specs['min_lr'])

    # Initialize session data / resume old session
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

    # Main loop
    while True:
        # Update epsilon
        epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                    episode_idx / specs['epsilon_decay_last_episode'])

        # Play one step
        res = agent.play_step(net, epsilon=epsilon)

        # End of an episode
        if res is not None:  
            score, max_tile, move_count = res

            # Time episode
            episode_duration = time.time() - episode_start
            writer.add_scalar("episode duration", episode_duration, episode_idx)

            # Log average of statistics
            writer.add_scalar("score", score, episode_idx)
            writer.add_scalar("max tile", max_tile, episode_idx)
            writer.add_scalar("epsilon", epsilon, episode_idx)
            writer.add_scalar("move count", move_count, episode_idx)

            # Testing
            if episode_idx % specs['test_freq'] == 0:
                m_test_score = test_greedy(agent, specs['test_size'], net, writer)
            
                # Save improved model
                if args.save_model and (best_test_score is None or best_test_score < m_test_score):
                    # Save model and related session data to be able to continue training from this point
                    session_data = {'episode_idx': episode_idx, 'step_idx': step_idx}
                    save_model(net, session_data, colab=args.colab, drive_dir=drive_dir)

                    best_test_score = m_test_score


            episode_idx += 1
            episode_start = time.time()

            # Update target net parameters
            # for target_param, param in zip(tgt_net.parameters(), net.parameters()):
            #     target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + \
            #                                 param.data * specs['tau'])

            if episode_idx % specs['sync_target_net_freq'] == 0:
                tgt_net.load_state_dict(net.state_dict())
        
        if args.colab and step_idx % 2000 == 0:
            copy_logs('runs', log_dir, drive_dir)
                

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