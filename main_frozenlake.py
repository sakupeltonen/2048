import argparse
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from datetime import datetime

from experience_replay import ExperienceBuffer
from dqn_model import DQN
from dqn_agent import DQNAgent
from environment import OnehotWrapper
from tools import visualize_qval

from tensorboardX import SummaryWriter




def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]

        # Double Q-learning 
        # next_state_actions = net(next_states_v).max(1)[1]
        # next_state_values = tgt_net(next_states_v).gather(1, 
        # next_state_actions.unsqueeze(-1)).squeeze(-1)
        
        
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
    args = parser.parse_args()

    specs_file = f"specs/{args.agent_name}.json"

    device = torch.device("cuda" if args.cuda else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, specs_file)
    specs = json.load(open(path, "r"))

    desc = specs['desc']
    # with this map, perfect winrate is possible
    width = len(desc[0])
    height = len(desc)

    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
    env = OnehotWrapper(env)
    maxval = 1
    num_actions = 4

    net = DQN(maxval, height, width, num_actions).to(device)
    tgt_net = DQN(maxval, height, width, num_actions).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    writer = SummaryWriter(comment=f"-{args.agent_name}")
    print(net)

    buffer = ExperienceBuffer(specs['replay_size'])
    agent = DQNAgent(env, buffer, device=device)

    epsilon = specs['epsilon_start']
    optimizer = optim.Adam(net.parameters(), lr=specs['learning_rate'])

    scores = []
    max_tiles = []
    episode_durations = []  # time in seconds
    move_counts = []  # steps
    losses = []
    stats_period = 500  # logged statistics averaged over stats_period

    episode_idx = 0
    episode_start = time.time()
    best_test_score = None

    while True:
        res = agent.play_step(net, epsilon=epsilon)  # FIX in main. epsilon is somehow passed even though keyword 
        

        # End of an episode
        if res is not None:  
            score, move_count = res

            # TODO simplify logging
            scores.append(score)
            move_counts.append(move_count)
            
            # Time episode
            episode_duration = time.time() - episode_start
            episode_start = time.time()
            episode_durations.append(episode_duration)

            # Log average of statistics
            if episode_idx >= stats_period:
                writer.add_scalar("mean score", np.mean(scores[-stats_period:]), episode_idx)
                writer.add_scalar("epsilon", epsilon, episode_idx)
                writer.add_scalar("mean move count", np.mean(move_counts[-stats_period:]), episode_idx)
                avg_loss = torch.mean(torch.tensor(losses[-stats_period:]))
                writer.add_scalar("average loss", avg_loss.item(), episode_idx)

            # Update epsilon
            epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                      episode_idx / specs['epsilon_decay_last_episode'])
            
            # Testing  # TODO agent can theoretically get stuck in cycles
            if episode_idx % specs['test_freq'] == 0:
                test_scores = []
                for i in range(specs['test_size']):
                    # Play a game with epsilon=0
                    while True:
                        res = agent.play_step(net)
                        if res:
                            score, move_count = res
                            test_scores.append(score)
                            break
                
                # Print and log test statistics
                m_test_score = np.mean(test_scores)

                print(f'Episode {episode_idx}: average score {m_test_score:.2f}')
                writer.add_scalar("greedy Test Score", 
                                  m_test_score, episode_idx // specs['test_freq'])
                
                current_time = datetime.now().strftime("%d%m-%H%M")
                visualize_qval(net.fc[0], width, height, title=f'{args.agent_name}-{current_time}-{episode_idx}')
            
                # Save improved model
                # if best_test_score is None or best_test_score < m_test_score:
                #     torch.save(net.state_dict(), "models/-best_%.0f.dat" % m_test_score)
                #     if best_test_score is not None:
                #         print("Best score updated %.3f -> %.3f" % (
                #             best_test_score, m_test_score))
                #     best_test_score = m_test_score

            episode_idx += 1

            
            # if episode_idx % specs['sync_target_net_freq'] == 0:
            #     tgt_net.load_state_dict(net.state_dict())
        
        # Update target net parameters  
        # # TODO move updating outside of end of episode loop in master
        for target_param, param in zip(tgt_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + \
                                        param.data * specs['tau'])

        if len(buffer) < specs['replay_start_size']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(specs['batch_size'])
        loss_t = calc_loss(batch, net, tgt_net, specs['gamma'], device=device)
        loss_t.backward()
        losses.append(loss_t.detach())
        optimizer.step()