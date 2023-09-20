import argparse
import pygame as pg
import json
import os
import pickle

from environment import Env2048
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
    env = Env2048(width=args.width, height=args.height, prob_2=args.prob_2, render_mode='human')
    _ = env.reset()

    history = [env.board.copy()]

    pg.init()
    screen = pg.display.set_mode((args.width * constants["boxsize"], 
                                  args.height * constants["boxsize"]))
    pg.display.set_caption("2048")
    pg_running = True
    terminated = False

    env.screen = screen
    env.render()

    def handle_keypress(direction):
        move = Env2048.action_to_int[direction]
        board, _, terminated, info = env.step(move)
        
        if not info['valid_move']:
            print(f'{direction} is not a valid move')
        else:
            history.append(board.copy())
            env.render()
            print(f'{direction} -- total score {env.score}')

            if terminated:
                print(f"Game over. Total score: {env.score}")

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", default=4, type=int)
    parser.add_argument("--height", default=4, type=int)
    parser.add_argument("--prob-2", default=0.9, type=float)
    parser.add_argument("--agent-name", default=None, type=str)
    parser.add_argument("--net_path", default=None, type=str)
    args = parser.parse_args()

    # specs_file = f"specs/{args.agent_name}.json"
    # net = DQN.from_file(args.net_path)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(script_dir, specs_file)
    # specs = json.load(open(path, "r"))

    human_play(args)
