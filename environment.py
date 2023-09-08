import numpy as np
import pygame
import json
import os
import gymnasium as gym
from gymnasium import spaces

script_dir = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(script_dir,'constants.json')
constants = json.load(open(constants_path, "r"))
theme = 'light'
# my_font = pygame.font.SysFont(constants["font"], constants["font_size"], bold=True)


temp = [2**i for i in range(14)]
log2 = {temp[i]: i for i in range(1,14)}
log2[0] = 0


class OnehotWrapper(gym.ObservationWrapper):
    """ Convert observation (board) to array of one-hot vectors """

    def __init__(self, env):
        super(OnehotWrapper, self).__init__(env)
    
    def to_onehot(self):
        n = log2[self.env.max_tile] + 1
        onehot = np.zeros((*self.env.board.shape, n), dtype=np.bool_)

        rows, cols = np.indices(self.env.board.shape)
        log_board = np.vectorize(log2.get)(self.env.board)
        onehot[rows, cols, log_board] = 1
        return onehot
    
    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        return self.to_onehot()
    
    def step(self, move, **kwargs):
        _, reward, done, info = self.env.step(move, **kwargs)
        _board = self.to_onehot()
        return _board, reward, done, info
        

class AfterstateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AfterstateWrapper, self).__init__(env) 

    # TODO not sure if want to do something with the intial state, where two tiles are spawned

    def step(self, move, **kwargs):
        _board, reward, done, info = self.env.step(move, **kwargs)
        board_copy = _board.copy()
        if info['valid_move']:
            c = info['spawn_location']
            board_copy[c[0]][c[1]] = 0
        
        # TODO should also return the final result
        return board_copy, reward, done, info


class Env2048(gym.Env):
    metadata = {"render_modes": ["ansi","human"]}

    action_to_int = {
        'down': 0,
        'left': 1,
        'up':  2,
        'right': 3
    }

    def __init__(self, width=4, height=4, prob_2=0.9, max_tile=4096, render_mode='ansi', screen=None):
        self.width = width
        self.height = height
        self.prob_2 = prob_2
        self.max_tile = max_tile
        self.observation_space = spaces.Box(low=0,
                                            high=max_tile,
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
            self._place_random_tile()
            self._place_random_tile()
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
    
    def _sample_tile(self):
        if np.random.random() < self.prob_2:
            return 2
        else:
            return 4

    def _place_random_tile(self):
        """ Place a random tile on board. Return the coordinate where the tile was placed. 
        Assumes that the board is not full."""
        zero_coords = np.argwhere(self.board == 0)
        assert len(zero_coords) > 0
        random_index = np.random.randint(0,len(zero_coords))
        c = zero_coords[random_index]
        self.board[c[0]][c[1]] = self._sample_tile()
        return c
        
    
    def is_done(self):
        zero_coords = np.argwhere(self.board == 0)
        if len(zero_coords) > 0:
            return False
        
        def exists_mergeable(_board):
            # Tests if two vertically adjacent tiles can be combined on _board
            height = len(_board)
            width = len(_board[0])
            
            for col in range(width):
                for row in range(1,height):
                    if _board[row-1][col] == _board[row][col]:
                        return True
            return False

        board_rotated = np.rot90(self.board)  
        # returns a rotated view of the original data. do not modify board_rotated!
        return not exists_mergeable(self.board) and not exists_mergeable(board_rotated)
    


    def step(self, move, generate=True):
        # assert isinstance(move, int)
        
        old_board = self.board.copy()  # compare with this to see if move is legal
        
        # rotated view of the original array
        self.board = np.rot90(self.board, k=move)

        # move tiles down in the rotated view
        reward = self._move_down()

        # reverse the rotation
        self.board = np.rot90(self.board, k=4-move)

        # move is legal if at least one tile moved
        valid_move = not np.array_equal(self.board, old_board)

        info = {'valid_move': valid_move}
        if valid_move and generate: 
            loc = self._place_random_tile()
            info['spawn_location'] = loc
        else:
            info['spawn_location'] = None

        done = self.is_done()

        return self.board, reward, done, info

    
    def _move_down(self):
        height = len(self.board)
        width = len(self.board[0])

        reward = 0
        # Handle each column independently
        for col in range(width):
            target_row = height - 1

            # moving row gets values height-2 to 0. Maintain that moving_row < target_row. 
            for moving_row in reversed(range(height - 1)):
                # nothing to move
                if self.board[moving_row][col] == 0:
                    continue
                # target row is empty. move non-zero value there
                elif self.board[target_row][col] == 0:
                    self.board[target_row][col] = self.board[moving_row][col]
                    self.board[moving_row][col] = 0
                # target and moving row non-empty
                else:
                    # tiles can be combined
                    if self.board[target_row][col] == self.board[moving_row][col]:
                        reward += self.board[target_row][col]
                        self.board[target_row][col] *= 2
                        self.board[moving_row][col] = 0
                        target_row -= 1
                    # tiles can't be combined. slide moving tile down
                    else:
                        # store value of moving tile, in case moving_row == target_row-1
                        val = self.board[moving_row][col]
                        self.board[moving_row][col] = 0
                        self.board[target_row-1][col] = val
                        target_row -=1
        return reward

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
        env._place_random_tile()

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