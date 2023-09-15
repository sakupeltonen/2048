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
        # next_state_values = tgt_net(next_states_v).max(1)[0]

        # Double Q-learning 
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        
        
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + \
                                   rewards_v
    loss = nn.MSELoss()(state_action_values,
                        expected_state_action_values)
    
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default="DQN-4x4")
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()

    args.agent_name = 'DQN-1x3'    
    specs_file = f"specs/{args.agent_name}.json"

    device = torch.device("cuda" if args.cuda else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, specs_file)
    specs = json.load(open(path, "r"))

    env = Env2048(width=specs['width'], height=specs['height'], prob_2=specs['prob_2'], max_tile=specs['max_tile'])
    env = OnehotWrapper(env)

    maxval = log2[specs['max_tile']] + 1
    net = DQN(maxval, specs['height'], specs['width'], specs['layer_size'], 4).to(device)
    tgt_net = DQN(maxval, specs['height'], specs['width'], specs['layer_size'], 4).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    writer = SummaryWriter(comment=f"-{args.agent_name}")
    print(net)

    buffer = ExperienceBuffer(specs['replay_size'])
    agent = DQNAgent(env, buffer)
    epsilon = specs['epsilon_start']
    optimizer = optim.Adam(net.parameters(), lr=specs['learning_rate'])

    scores = []
    max_tiles = []
    episode_durations = []  # time in seconds
    move_counts = []  # steps
    # frac_losses = []
    losses = []
    stats_period = 500  # 

    episode_idx = 0
    episode_start = time.time()
    best_test_score = None

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
                # writer.add_scalar("mean episode duration", np.mean(episode_durations[-stats_period:]), episode_idx)
                writer.add_scalar("mean move count", np.mean(move_counts[-stats_period:]), episode_idx)

                # writer.add_scalar("average fractional loss", np.mean(frac_losses[-stats_period:]), episode_idx)
                avg_loss = torch.mean(torch.tensor(losses[-stats_period:]))
                writer.add_scalar("average loss", avg_loss.item(), episode_idx)

            epsilon = max(specs['epsilon_final'], specs['epsilon_start'] -
                      episode_idx / specs['epsilon_decay_last_episode'])
            
            if episode_idx % specs['test_freq'] == 0:
                test_scores = []
                test_max_tiles = []
                invalid_counts = []
                for i in range(specs['test_size']):
                    while True:
                        res = agent.play_step(net, device=device)  # epsilon=0
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
            
                if best_test_score is None or best_test_score < m_test_score:
                    torch.save(net.state_dict(), "models/-best_%.0f.dat" % m_test_score)
                    if best_test_score is not None:
                        print("Best score updated %.3f -> %.3f" % (
                            best_test_score, m_test_score))
                    best_test_score = m_test_score

            episode_idx += 1



            # if episode_idx % specs['sync_target_net_freq'] == 0:
            #     tgt_net.load_state_dict(net.state_dict())

            
            for target_param, param in zip(tgt_net.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - specs['tau']) + param.data * specs['tau'])
            

        if len(buffer) < specs['replay_start_size']:
            continue
        

        optimizer.zero_grad()
        batch = buffer.sample(specs['batch_size'])
        loss_t = calc_loss(batch, net, tgt_net, specs['gamma'], device=device)
        loss_t.backward()
        losses.append(loss_t.detach())
        # frac_losses.append(frac_loss)
        optimizer.step()

    # writer.close()