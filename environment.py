import numpy as np
# import pygame

import gymnasium as gym
from gymnasium import spaces




class Env2048(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    PROB_2 = 0.9

    action_to_int = {
        'down': 0,
        'left': 1,
        'up':  2,
        'right': 3
    }

    def __init__(self, render_mode='ansi', width=4, height=4, empty=False):
        self.width = width
        self.height = height

        self.observation_space = spaces.Box(low=0,
                                            high=2**16,
                                            shape=(self.width, self.height),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.board = np.zeros((self.height, self.width), dtype=np.int32)
        if not empty:
            self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int32)
        self._place_random_tile()
        self._place_random_tile()
        return self.board
    
    def render(self):
        if self.render_mode == 'ansi':
            for row in self.board:
                print(' \t'.join(map(str, row)))

    def close(self):
        pass
    
    def _sample_tile(self):
        if np.random.random() < Env2048.PROB_2:
            return 2
        else:
            return 4

    def _place_random_tile(self):
        zero_coords = np.argwhere(self.board == 0)
        if len(zero_coords) > 0:
            random_index = np.random.randint(0,len(zero_coords))
            c = zero_coords[random_index]
            self.board[c[0]][c[1]] = self._sample_tile()
    
    def is_done(self):
        zero_coords = np.argwhere(self.board == 0)
        if len(zero_coords) > 0:
            return False
        
        def exists_mergeable(board):
            # Tests if two vertically adjacent tiles can be combined on board
            for col in range(self.width):
                for row in range(1,self.height):
                    if board[row-1][col] == board[row][col]:
                        return True
            return False

        board_rotated = np.rot90(self.board)  # TODO make sure this does not modify in place
        return not exists_mergeable(self.board) and not exists_mergeable(board_rotated)


    def step(self, move, verbose=False):
        if isinstance(move, str):
            move = Env2048.action_to_int[move]
        
        board_copy = self.board.copy()

        board_rotated = np.rot90(self.board, k=move)
        board_updated, reward = self._move_down(board_rotated)
        self.board = np.rot90(board_updated, k=4-move)

        board_changed = not np.array_equal(board_copy, self.board)
        if board_changed:
            self._place_random_tile()

        done = self.is_done()

        if verbose:
            movestr = {0: 'down', 1: 'left', 2: 'up', 3: 'right'}[move]
            print(f"move: {movestr}")
            self.render()

        return self.board, reward, done, {}
    

    def _move_down(self, board):
        reward = 0
        # Handle each column independently
        for col in range(self.width):
            target_row = self.height - 1

            # moving row gets values height-2 to 0. Maintain that moving_row < target_row. 
            for moving_row in reversed(range(self.height - 1)):
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
    def random_state(cls, max_power=6, n_tiles=8):
        env = Env2048(empty=True)
        
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
    def custom_state(cls, tiles, width=4, height=4):
        board = np.zeros((height, width), dtype=np.int32)
        i = 0
        for y in range(height):
            for x in range(width):
                board[y][x] = tiles[i]
                i += 1
        env = Env2048(empty=True, width=width, height=height)
        env.board = board
        env.render()
        return env
