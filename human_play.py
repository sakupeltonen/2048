import argparse
import pygame as pg
import json
import os
import pickle
from environment import Env2048
from tools import save_game

script_dir = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(script_dir,'constants.json')
constants = json.load(open(constants_path, "r"))


def pg_main(width, height, initialize, handle_keypress):
    """
       Controller for pygame. Behavior is determined by the functions initialize and handle_keypress.
    """
    pg.init()
    screen = pg.display.set_mode((width * constants["boxsize"], 
                                  height * constants["boxsize"]))
    pg.display.set_caption("2048")
    pg_running = True
    terminated = False

    initialize(screen)

    direction_from_pg = {
        pg.K_LEFT: 'left', 
        pg.K_RIGHT: 'right',
        pg.K_UP: 'up',
        pg.K_DOWN: 'down'
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


def human_play(width=4, height=4, prob_2=0.9, initial_state=None, filename=None):
    env = Env2048(width=width, height=height, prob_2=prob_2, render_mode='human')
    _ = env.reset(custom_state=initial_state)

    history = [env.board.copy()]
    game_score = 0

    def initialize(screen):
        env.screen = screen
        env.render()

    def handle_keypress(direction):
        nonlocal game_score
        move = Env2048.action_to_int[direction]
        board, reward, terminated, info = env.step(move)
        game_score += reward
        if not info['valid_move']:
            print(f'{direction} is not a valid move')
        else:
            history.append(board.copy())
            env.render()
            print(f'{direction} -- total score {game_score}')
            #env.render(mode='ansi')

            if terminated:
                print(f"Game over. Total score: {game_score}")

    pg_main(width, height, initialize, handle_keypress)
    if filename:
        save_game(history, filename)



def replay_game(path):
    """
       Replay a saved game. 
       
       Left/Right arrows go forwards/backwards by one step. 
       Up/Down by 10 steps. 
    """
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

        if direction == 'up':
            state_index += 10
            state_index = min(state_index, len(history)-1)

        if direction == 'down':
            state_index -= 10
            state_index = max(state_index, 0)

        env.board = history[state_index]
        env.render()

    pg_main(width, height, initialize, handle_keypress)



# replay_game('games/testi.pkl')  # TODO implement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", default=4, type=int)
    parser.add_argument("--height", default=4, type=int)
    parser.add_argument("--prob-2", default=0.9, type=float)
    parser.add_argument("--save", default=None, type=str)
    args = parser.parse_args()

    human_play(width=args.width, height=args.height, prob_2=args.prob_2, filename=args.save)
