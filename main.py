import numpy as np
import pygame as pg
import json
from enum import Enum
import gymnasium as gym
from environment import Env2048
from greedy import GreedyAgent

np.random.seed(42)

constants = json.load(open("constants.json", "r"))

direction_from_pg = {
    pg.K_LEFT: 'left',
    pg.K_RIGHT: 'right',
    pg.K_UP: 'up',
    pg.K_DOWN: 'down'
}



pg.init()
screen = pg.display.set_mode((constants["size"], constants["size"]))
pg.display.set_caption("2048")

env = Env2048(render_mode='human')
_ = env.reset()
# env = Env2048.custom_state([16,128,16,128,64,8,64,8,256,1024,256,1024,32,64,0,0],render_mode='human')
env.screen = screen
env.render()

history = [env.board.copy()]

greedy = GreedyAgent()
player = 'AI'
pg_running = True
terminated = False
score = 0
while pg_running:
    
    direction = None
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg_running = False
        
        if not terminated:

            if event.type == pg.KEYDOWN:
                if player == 'human':
                    if event.key in direction_from_pg.keys():
                        direction = direction_from_pg[event.key]
                else:
                    direction = greedy.search(env, depth=5)
            #pg.time.delay(2000)

        if direction is not None:
            board, reward, terminated, _ = env.step(direction)
            score += reward
            history.append(board.copy())
            env.render()
            print(direction)
            env.render(mode='ansi')

            if terminated:
                print(f"Game over. Total score: {score}")
            
            direction = None


pg.quit()

# TODO rewrite game loop.
# already implemented in gym. Not sure how we can pass the screen to the environment
# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
# play(gym.make("CartPole-v0"), keys_to_action=mapping)