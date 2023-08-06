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


# a = Env2048.custom_state([4,2,4,2,2,4,2,4,4,2,4,2,2,4,2,4])

pg.init()
screen = pg.display.set_mode((constants["size"], constants["size"]))
pg.display.set_caption("2048")

env = Env2048(render_mode='human')
env.screen = screen
observation = env.reset()
env.render()

history = [env.board.copy()]

mrGreedy = GreedyAgent()
player = 'AI'
running = True
while running:
    
    direction = None
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        
        if event.type == pg.KEYDOWN:
            if player == 'human':
                if event.key in direction_from_pg.keys():
                    direction = direction_from_pg[event.key]
            else:
                direction = mrGreedy.move(env)
        #pg.time.delay(2000)

    if direction is not None:
        board, reward, terminated, _ = env.step(direction)
        history.append(board.copy())
        print('')
        env.render()


pg.quit()

# TODO apparently already implemented. Not sure how we can pass the screen to the environment
# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
# play(gym.make("CartPole-v0"), keys_to_action=mapping)