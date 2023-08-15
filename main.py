import numpy as np
import pygame as pg
import json
from enum import Enum
import gymnasium as gym
import os
import pickle
from environment import Env2048
from greedy import GreedyAgent
from tools import save_game

np.random.seed(42)

constants = json.load(open("constants.json", "r"))

direction_from_pg = {
    pg.K_LEFT: 'left', 
    pg.K_RIGHT: 'right',
    pg.K_UP: 'up',
    pg.K_DOWN: 'down'
}



def pg_main(width, height, initialize, handle_keypress):
    pg.init()
    screen = pg.display.set_mode((width * constants["boxsize"], height * constants["boxsize"]))
    pg.display.set_caption("2048")
    pg_running = True
    terminated = False

    initialize(screen)

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







def human_play(width=4, height=4, initial_state=None):
    env = Env2048(width=width, height=height, render_mode='human')
    _ = env.reset(custom_state=initial_state)
    # _ = env.reset(custom_state=[16,128,16,128,64,8,64,8,256,1024,256,1024,32,64,0,0])

    history = [env.board.copy()]
    game_score = 0  # TODO fix score not being updated, outside of the scope of handle_keypress

    def initialize(screen):
        env.screen = screen
        env.render()

    def handle_keypress(direction):
        nonlocal game_score
        board, reward, terminated, info = env.step(direction)
        game_score += reward
        if not info['valid_move']:
            print(f'{direction} is not a valid move')
        else:
            #score += reward
            history.append(board.copy())
            env.render()
            print(f'{direction} -- total score {game_score}')
            env.render(mode='ansi')

            if terminated:
                print(f"Game over. Total score: {game_score}")

    pg_main(width, height, initialize, handle_keypress)
    save_game(history)



def replay_game(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)

    state_index = 0
    board = history[state_index]
    width = len(board[0])
    height = len(board)
    env = Env2048(width=width, height=height, render_mode='human')
    _ = env.reset(custom_state=board)

    def initialize(screen):
        env.screen = screen
        env.render()

    def handle_keypress(direction):
        nonlocal state_index
        if direction == 'left':
            if state_index > 0:
                state_index -= 1

        if direction == 'right':
            if state_index < len(history) - 1:
                state_index += 1

        env.board = history[state_index]
        env.render()

    pg_main(width, height, initialize, handle_keypress)


human_play(width=3, height=2)
# replay_game('games/128bad.pkl')