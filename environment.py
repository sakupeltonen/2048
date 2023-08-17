import numpy as np
import pygame
import json

import gymnasium as gym
from gymnasium import spaces

constants = json.load(open("constants.json", "r"))
theme = 'light'
# my_font = pygame.font.SysFont(constants["font"], constants["font_size"], bold=True)


class Env2048(gym.Env):
    metadata = {"render_modes": ["ansi","human"]}

    PROB_2 = 0.9  
    # PROB_2 = 1  # TEMP

    action_to_int = {
        'down': 0,
        'left': 1,
        'up':  2,
        'right': 3
    }

    def __init__(self, width=4, height=4, render_mode='ansi', screen=None):
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0,
                                            high=2**16,
                                            shape=(height, width),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = screen

        self.board = np.zeros((height, width), dtype=np.int32)
        self.score = 0
        

    def reset(self, empty=False, custom_state=None):
        if empty: 
            return self.board
        if custom_state is None:
            self.board = np.zeros((self.height, self.width), dtype=np.int32)
            self.board = Env2048._place_random_tile(self.board)
            self.board = Env2048._place_random_tile(self.board)
        else:
            # custom_state possibly given as a one-dimensional list (in row-major order), or 
            # as a two-dimensional np.array
            if isinstance(custom_state, list):
                assert len(custom_state) == self.height * self.width
                i = 0
                for y in range(self.height):
                    for x in range(self.width):
                        self.board[y][x] = custom_state[i]
                        i += 1
            else:
                self.board = custom_state

        return self.board
    

    def render(self,mode=None):
        if mode is None: 
            mode = self.render_mode

        if mode == 'ansi':
            for row in self.board:
                print(' \t'.join(map(str, row)))
            print("")

        if mode == 'human':
            self._render_pygame()

    def close(self):
        pass
    
    @staticmethod
    def _sample_tile():
        if np.random.random() < Env2048.PROB_2:
            return 2
        else:
            return 4

    @staticmethod
    def _place_random_tile(board):
        zero_coords = np.argwhere(board == 0)
        res = board.copy()
        if len(zero_coords) > 0:
            random_index = np.random.randint(0,len(zero_coords))
            c = zero_coords[random_index]
            res[c[0]][c[1]] = Env2048._sample_tile()
        return res
    
    @staticmethod
    def is_done(board):
        zero_coords = np.argwhere(board == 0)
        if len(zero_coords) > 0:
            return False
        
        def exists_mergeable(board):
            height = len(board)
            width = len(board[0])
            # Tests if two vertically adjacent tiles can be combined on board
            for col in range(width):
                for row in range(1,height):
                    if board[row-1][col] == board[row][col]:
                        return True
            return False

        board_rotated = np.rot90(board)
        return not exists_mergeable(board) and not exists_mergeable(board_rotated)

    @staticmethod
    def _step(board, move, generate=True):
        if isinstance(move, str):
            move = Env2048.action_to_int[move]

        board_rotated = np.rot90(board.copy(), k=move)  # note that the rotated board is a view of the original array (still linked)
        
        board_updated, reward = Env2048._move_down(board_rotated)
        afterstate = np.rot90(board_updated, k=4-move)

        # directions that don't slide/combine any tiles are not valid (but also don't give any points)
        valid_move = not np.array_equal(board, afterstate)
        # A new tile is generated if the move is valid. generate=False turns off new tile generation
        if valid_move and generate:
            board_result = Env2048._place_random_tile(afterstate)
        else:
            board_result = afterstate
        info = {'valid_move': valid_move}
        info['afterstate'] = afterstate

        done = Env2048.is_done(board_result)

        return board_result, reward, done, info
    
    def step(self, move):
        board, reward, done, info = Env2048._step(self.board, move)
        self.board = board
        self.score += reward
        return board, reward, done, info
    
    @staticmethod
    def _move_down(board):
        height = len(board)
        width = len(board[0])

        reward = 0
        # Handle each column independently
        for col in range(width):
            target_row = height - 1

            # moving row gets values height-2 to 0. Maintain that moving_row < target_row. 
            for moving_row in reversed(range(height - 1)):
                # nothing to move
                if board[moving_row][col] == 0:
                    continue
                # target row is empty. move non-zero value there
                elif board[target_row][col] == 0:
                    board[target_row][col] = board[moving_row][col]
                    board[moving_row][col] = 0
                # target and moving row non-empty
                else:
                    # tiles can be combined
                    if board[target_row][col] == board[moving_row][col]:
                        reward += board[target_row][col]
                        board[target_row][col] *= 2
                        board[moving_row][col] = 0
                        target_row -= 1
                    # tiles can't be combined. slide moving tile down
                    else:
                        # store value of moving tile, in case moving_row == target_row-1
                        val = board[moving_row][col]
                        board[moving_row][col] = 0
                        board[target_row-1][col] = val
                        target_row -=1
        return board, reward

    @classmethod
    def random_state(cls, width=4, height=4, max_power=6, n_tiles=8,render_mode='ansi'):
        env = Env2048(width=width, height=height)
        
        all_coords = [(y,x) for y in range(env.height) for x in range(env.width)]  # get list of all coordinates
        random_subset_indices = np.random.choice(len(all_coords), n_tiles, replace=False)
        random_coords = [all_coords[idx] for idx in random_subset_indices]

        tiles = [2**x for x in range(1,max_power+1)]
        tiles = np.random.choice(tiles, size=n_tiles, replace=True)

        for i in range(n_tiles):
            c = random_coords[i]
            env.board[c[0]][c[1]] = tiles[i]

        return env
    
    @classmethod 
    def initial_with_tile(cls, val, width=4, height=4, render_mode='human'):
        env = Env2048(width=width, height=height, render_mode=render_mode)

        all_coords = [(y,x) for y in range(env.height) for x in range(env.width)]  # get list of all coordinates
        random_index = np.random.choice(len(all_coords), 1)
        c = [all_coords[idx] for idx in random_index][0]

        env.board[c[0]][c[1]] = val
        env.board = Env2048._place_random_tile(env.board)

        return env
    
    def _render_pygame(self):
        """
        Credits to github.com/rajitbanerjee/2048-pygame/
        """
        assert self.screen is not None
        
        my_font = pygame.font.SysFont(constants["font"], constants["font_size"], bold=True)
        
        self.screen.fill(tuple(constants["colour"][theme]["background"]))
        box_size = constants["boxsize"]
        padding = constants["padding"]
        for i in range(self.height):
            for j in range(self.width):
                colour = tuple(constants["colour"][theme][str(self.board[i][j])])
                pygame.draw.rect(self.screen, colour, (j * box_size + padding,
                                                i * box_size + padding,
                                                box_size - 2 * padding,
                                                box_size - 2 * padding), 0)
                if self.board[i][j] != 0:
                    if self.board[i][j] in (2, 4):
                        text_colour = tuple(constants["colour"][theme]["dark"])
                    else:
                        text_colour = tuple(constants["colour"][theme]["light"])
                    # display the number at the centre of the tile
                    self.screen.blit(my_font.render("{:>4}".format(
                        self.board[i][j]), 1, text_colour),
                        (j * box_size + 2.5 * padding, i * box_size + 7 * padding))
        pygame.display.update()