import numpy as np
from enum import Enum
import gymnasium as gym
from environment import Env2048

np.random.seed(42)

direction_from_keyboard = {
    's': 'down',
    'd': 'right', 
    'w': 'up',
    'a': 'left'
}

env = Env2048()
observation = env.reset()
env.render()

# a = Env2048.random_state()
# print("")
# a.render()
# a.step("down", verbose=True)
# a.step("right", verbose=True)
# a.step("left", verbose=True)
# a.step("up", verbose=True)

# a = Env2048.custom_state([4,2,4,2,2,4,2,4,4,2,4,2,2,4,2,4])


while not env.is_done():
    user_input = input()
    if user_input == "q":
        break
    elif user_input in direction_from_keyboard.keys():
        direction = direction_from_keyboard[user_input]
    else:
        print(f'{user_input} is not a valid input')

    board, reward, terminated, _ = env.step(direction)
    print('')
    env.render()

env.close()