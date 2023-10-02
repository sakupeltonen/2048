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


class PenalizeMovingUpWrapper(gym.Wrapper):
    def __init__(self, env, up_penalty_factor):
        super(PenalizeMovingUpWrapper, self).__init__(env)
        self.up_penalty_factor = up_penalty_factor

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, move, **kwargs):
        observation, reward, done, info = self.env.step(move, **kwargs)
        
        if move == 2:
            reward *= self.up_penalty_factor
            
        return observation, reward, done, info
    

    def available_moves():
        '''TEMP'''
        res = self.env.unwrapped.available_moves()
        if len(res) > 1 and 2 in res:
            res.remove(2)
        return res


class NextStateWrapper(gym.ObservationWrapper):
    '''Give reward and possible next state for each move'''
    def __init__(self, env):
        super(NextStateWrapper, self).__init__(env)
        self.copy_env = Env2048(width=env.unwrapped.width, height=env.unwrapped.height, 
                           prob_2=env.unwrapped.prob_2, max_tile=env.unwrapped.max_tile)
        # self.copy_env = OnehotWrapper(copy_env)
    
    def simulate_moves(self, _board, debug=False):
        # _board can be specified for debugging purposes
        res = []
        for move in range(self.env.action_space.n):
            self.copy_env.board = _board.copy()
            board, reward, done, info = self.copy_env.step(move)

            # reward is approximately normalized. it can be higher than max_tile in rare cases
            reward_norm = reward / self.env.unwrapped.max_tile

            # x = np.concatenate((board.flatten(), 
            #                     np.array([reward_norm, done, info['valid_move']])))
            x = np.array([reward_norm, done, info['valid_move']])
            if debug:
                print(f"({move}) reward: {reward}, done: {done}, valid: {info['valid_move']}")
            res.append(x)
        return np.concatenate(res)
    
    def reset(self, **kwargs):
        board = self.env.reset(**kwargs)
        res = self.simulate_moves(self.env.unwrapped.board)
        return np.concatenate((board.flatten(), res))
    
    def step(self, move, **kwargs):
        board, reward, done, info = self.env.step(move, **kwargs)
        res = self.simulate_moves(self.env.unwrapped.board)
        
        obs_v = np.concatenate((board.flatten(), res))
        return obs_v, reward, done, info

    def observation(self):
        # Temporary solution to get observation of a manually updated board (for debugging), when not actually moving.
        # Overlaps with step method, but didn't want to deviate from gym environment customs
        obs = self.env.observation()
        additional_obs = self.simulate_moves(self.env.unwrapped.board)
        return np.concatenate((obs.flatten(), additional_obs))


def to_onehot(board, max_tile):
    n = log2[max_tile] + 1
    onehot = np.zeros((*board.shape, n), dtype=np.bool_)
    rows, cols = np.indices(board.shape)
    log_board = np.vectorize(log2.get)(board)
    onehot[rows, cols, log_board] = 1
    return onehot

class OnehotWrapper(gym.ObservationWrapper):
    """ Convert observation (board) to array of one-hot vectors """

    def __init__(self, env):
        super(OnehotWrapper, self).__init__(env)
    
    def reset(self, **kwargs):
        _board = self.env.reset(**kwargs)
        return to_onehot(_board, self.env.unwrapped.max_tile)
    
    def step(self, move, **kwargs):
        _board, reward, done, info = self.env.step(move, **kwargs)
        onehot_board = to_onehot(_board, self.env.unwrapped.max_tile)
        return onehot_board, reward, done, info

    def observation(self):
        # TEMP solution to get the one-hot encoded board 
        return to_onehot(self.env.unwrapped.board, self.env.unwrapped.max_tile)
    

class RotationInvariantWrapper(gym.ObservationWrapper):
    """Return rotated board that minimizes a hash function value. The hash is not computed explicitly for efficiency"""
    # TODO could also add flips
    def __init__(self, env):
        super(RotationInvariantWrapper, self).__init__(env) 

    def minHash(self):
        rotated_boards = {i: np.rot90(self.env.unwrapped.board, i) for i in range(4)}  # 4 hardcoded

        candidates = list(range(4))
        if self.env.unwrapped.width == 1:
            candidates.remove(1) 
            candidates.remove(3)
        if self.env.unwrapped.height == 1:
            candidates.remove(0)
            candidates.remove(2)

        for y in reversed(range(self.env.unwrapped.height)):
            for x in reversed(range(self.env.unwrapped.width)):
                # get tile at [y][x] in all remaining rotations
                tiles = [rotated_boards[i][y][x] for i in candidates]
                
                # remaining candidates
                candidates = [i for i in candidates if rotated_boards[i][y][x] == max(tiles)]

                if len(candidates) == 1:
                    return candidates[0]
        
        # board may have rotational symmetries, in which case the first one is chosen
        return candidates[0]
                

    def reset(self, **kwargs):
        board = self.env.reset(**kwargs)
        rot = self.minHash()

        # store observed rotation 
        self.observed_rotation = rot 

        return np.rot90(board, rot)


    def step(self, move, **kwargs):
        rotated_move = (move + self.observed_rotation) % 4  #  4 hardcoded  # TODO check
        board, reward, done, info = self.env.step(rotated_move, **kwargs)
        
        # compute new rotation
        rot = self.minHash()
        # update observed rotation
        self.observed_rotation = rot

        return np.rot90(board, rot), reward, done, info   # TODO info['board'] might need to be rotated as well, depending on the order of the wrappers
    
    def available_moves(self):
        original = self.env.unwrapped.available_moves()

        return [original[(a + self.observed_rotation) % 4] for a in range(self.env.action_space.n)]  # TODO check
    
    def get_board(self):
        return np.rot90(self.env.unwrapped.board, self.observed_rotation), self.observed_rotation

    



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
        self.legal_move_count = 0
        

    def reset(self, empty=False, custom_state=None):
        self.score = 0
        self.legal_move_count = 0
        if empty: 
            return self.board
        if custom_state is None:
            self.board = np.zeros((self.height, self.width), dtype=np.int32)
            self.place_random_tile()
            self.place_random_tile()
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

    def place_random_tile(self):
        """ Place a random tile on board. Return the coordinate where the tile was placed. 
        Assumes that the board is not full."""
        zero_coords = np.argwhere(self.board == 0)
        assert len(zero_coords) > 0
        random_index = np.random.randint(0,len(zero_coords))
        c = zero_coords[random_index]
        self.board[c[0]][c[1]] = self._sample_tile()
        return c
        

    def available_moves(self):
        """Return valid moves on self.board"""
        def can_move_down(_board):
            """Test if down is a valid move on _board"""
            height = len(_board)
            width = len(_board[0])

            for col in range(width):
                for row in range(height-1):
                    if _board[row][col] == 0:
                        continue
                    mergeable = _board[row][col] == _board[row + 1][col]
                    moveable = _board[row + 1][col] == 0

                    if mergeable or moveable:
                        return True
            return False

        res = np.array([False]*self.action_space.n)
        for i in range(4):
            _board = np.rot90(self.board, i)
            res[i] = can_move_down(_board)
        return res

    
    def is_done(self):
        # TODO this might be slightly faster than using available_moves, but could combine for simplicity
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
        # rot90 returns a rotated view of the original data. do not modify board_rotated!

        return not exists_mergeable(self.board) and not exists_mergeable(board_rotated)
    


    def step(self, move, generate=True):
        # assert isinstance(move, int)
        
        old_board = self.board.copy()  # compare with this to see if move is legal
        
        # rotated view of the original array
        self.board = np.rot90(self.board, k=move)

        # move tiles down in the rotated view
        reward = self._move_down()

        self.score += reward

        # reverse the rotation
        self.board = np.rot90(self.board, k=4-move)

        # move is legal if at least one tile moved
        valid_move = not np.array_equal(self.board, old_board)

        info = {'valid_move': valid_move}
        if valid_move and generate: 
            loc = self.place_random_tile()
            info['spawn_location'] = loc
        else:
            info['spawn_location'] = None

        if valid_move:
            self.legal_move_count += 1

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
                        self.board[target_row][col] *= 2
                        reward += self.board[target_row][col]
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