import argparse
import pygame as pg
import json
import os
import pickle
import numpy as np
import torch

from environment import Env2048, OnehotWrapper, NextStateWrapper, log2
from tools import save_game
from dqn_model import DQN
from dqn_agent import DQNAgent

script_dir = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(script_dir,'constants.json')
constants = json.load(open(constants_path, "r"))

direction_from_pg = {
        pg.K_LEFT: 'left', 
        pg.K_RIGHT: 'right',
        pg.K_UP: 'up',
        pg.K_DOWN: 'down'
    }


def human_play(args):
    visualize_DQN = args.net_file is not None

    # dont use wrappers but get the observation manually. better for when manually changing board. no, actually write an observe method
    if not visualize_DQN:
        env = Env2048(width=args.width, height=args.height, prob_2=args.prob_2, render_mode='human')
    else:
        assert args.agent_name
        specs_file = f"specs/{args.agent_name}.json"
        path = os.path.join(script_dir, specs_file)
        specs = json.load(open(path, "r"))

        env = Env2048(width=specs['width'], height=specs['height'], prob_2=specs['prob_2'], max_tile=specs['max_tile'], render_mode='human')
        env = OnehotWrapper(env)
        env = NextStateWrapper(env)

        max_val = log2[specs['max_tile']] + 1
        device = torch.device('cpu')  # todo convert gpu tensors to cpu 

        net_args = (max_val, specs['height'], specs['width'], specs['layer_size'], 4)
        path = os.path.join(script_dir, args.net_file)
        net = DQN.from_file(args.net_file, device, *net_args)


    def print_qvals():
        obs = env.observation()
        obs_a = np.array([obs], copy=False)
        obs_v = torch.tensor(obs_a).to(device)
        q_vals = net(obs_v)[0]
        print(f"\ndown {q_vals[0].item():.3f}\tleft {q_vals[1].item():.3f}\tup {q_vals[2].item():.3f}\tright {q_vals[3].item():.3f}")
    

    _ = env.reset()
    if visualize_DQN: 
        print_qvals()
    history = [env.unwrapped.board.copy()]
    actions = []

    pg.init()
    screen = pg.display.set_mode((args.width * constants["boxsize"], 
                                  args.height * constants["boxsize"]))
    pg.display.set_caption("2048")
    pg_running = True
    terminated = False

    env.unwrapped.screen = screen
    env.unwrapped.render()


    def handle_keypress(direction):
        move = Env2048.action_to_int[direction]
        observation, _, terminated, info = env.step(move)
        board = env.unwrapped.board
        
        if not info['valid_move']:
            print(f'{direction} is not a valid move')
        else:
            history.append(board.copy())
            actions.append(move)
            env.unwrapped.render()
            if args.verbose:
                print(f'{direction} -- total score {env.unwrapped.score}')

            if terminated:
                print(f"Game over. Total score: {env.unwrapped.score}")
        
            if visualize_DQN:
                print_qvals()

    while pg_running:
        direction = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg_running = False
            
            if not terminated:
                if event.type == pg.KEYDOWN:
                    if event.key in direction_from_pg.keys():
                        direction = direction_from_pg[event.key]
                        handle_keypress(direction)
                    elif event.key == pg.K_a:
                        pass
                    elif event.key == pg.K_d:
                        pass
    pg.quit()
    save_game(history, actions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", default=4, type=int)
    parser.add_argument("--height", default=4, type=int)
    parser.add_argument("--prob-2", default=0.9, type=float)
    parser.add_argument("--net-file", default=None, type=str, help='Relative path to saved DQN model to be visualized')
    parser.add_argument("--agent-name", default=None, type=str, help='Agent name corresponding to the visualized DQN')
    parser.add_argument("--verbose", default=False, action="store_true", help='Print moves and score')
    args = parser.parse_args()

    # specs_file = f"specs/{args.agent_name}.json"
    # net = DQN.from_file(args.net_path)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(script_dir, specs_file)
    # specs = json.load(open(path, "r"))

    human_play(args)
