import argparse
import pygame as pg
import json
import os
import gymnasium as gym
from environment import Env2048
from tools import save_game

script_dir = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(script_dir,'constants.json')
constants = json.load(open(constants_path, "r"))


def pg_main(initialize, handle_keypress):
    """
       Controller for pygame. Behavior is determined by the functions initialize and handle_keypress.
    """
    pg.init()
    screen = pg.display.set_mode((4 * constants["boxsize"], 
                                  4 * constants["boxsize"]))
    pg_running = True
    terminated = False

    initialize(screen)

    direction_from_pg = {
        pg.K_LEFT: 0, 
        pg.K_RIGHT: 2,
        pg.K_UP: 3,
        pg.K_DOWN: 1
    }

    while pg_running:
        direction = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg_running = False
            
            if not terminated:
                if event.type == pg.KEYDOWN:
                    if event.key in direction_from_pg.keys():
                        direction = direction_from_pg[event.key]

            if direction is not None:
                handle_keypress(direction)
                
                direction = None

    pg.quit()


def human_play(desc):
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode='human')
    _ = env.reset()


    def initialize(screen):
        env.screen = screen
        env.render()

    def handle_keypress(direction):
        board, reward, terminated, _, info = env.step(direction)
        env.render()
        if terminated:
            print(f"Game over. Total score: {reward}")

    pg_main(initialize, handle_keypress)



if __name__ == "__main__":
    desc = ['SFFH','FFFF', 'FFFF','HFFG']

    human_play(desc)
