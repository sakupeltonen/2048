import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
import math
from environment import Env2048
from tools import save_game

np.random.seed(42)

temp = [2**i for i in range(12)]
log2 = {temp[i]: i for i in range(1,12)}
log2[0] = 0

MAX_VAL = 7
OPTIMISTIC_INIT = 2**MAX_VAL  # something in the ball park of a solid score

class NTuple:
    def __init__(self, coords, max_val, LUT=None, update_counts=None):
        """
           Parameters:
           coords ([int]): list of board coordinates (order is important)
           - n=len(indices) as in NTuple
           max_val (type): Maximum possible value in array
           - max_val^n is the size of the lookup table
           LUT, update_counts: pointers to arrays for other NTuples, when initialized as a symmetric copy
           TODO have to be very careful when computing the coords for symmetric tuples
        """
        self.coords = coords
        n = len(coords)
        self.n = n
        self.max_val = max_val
        size = (max_val+1) ** n
        self.size = size

        if LUT is None:
            self.LUT = np.zeros(size) + OPTIMISTIC_INIT  # TODO remove hardcoded optimistic initialization
        else:
            self.LUT = LUT

        if update_counts is None:
            self.update_counts = np.zeros(size) 
        else:
            self.update_counts = update_counts

    def index(self, board):
        """ Get index i in LUT corresponding to the tuple given by board[self.coords] """
        values = [board[i][j] for (i,j) in self.coords]
        values = [log2[v] for v in values]
        res = 0
        for i in range(self.n):
            res += values[i] * (self.max_val ** i)
        return res

    def evaluate(self, board):
        """ NTuple(board) returns the value of the NTuple for the given tuple at board[self.coords] """
        return self.LUT[self.index(board)]

    def update(self, board, difference):
        i = self.index(board)
        self.LUT[i] += difference
        self.update_counts[i] += 1


class NTupleNetwork:
    def __init__(self, all_coords, max_val):
        self.tuples = [] 
        self.init_tuples(all_coords, max_val)

    def init_tuples(self, all_coords, max_val):
        def reflect_horizontal(coords):
            return [(3-y,x) for (y,x) in coords]  # TODO remove hardcoded height 3
        
        def rotate_clockwise(coords):
            # e.g. (y,x)=(3,3) --> (3,0); (y,x)=(2,1) --> (1,1) 
            return [(x,3-y) for (y,x) in coords] # TODO remove hardcoded width 3

        
        for coords in all_coords:
            rotated_coords = [(y,x) for (y,x) in coords]
            reflected_rotated_coords = reflect_horizontal(coords)

            original_n_tuple = NTuple(rotated_coords, max_val)
            LUT = original_n_tuple.LUT
            self.tuples.append(original_n_tuple)

            copy_tuple = NTuple(reflected_rotated_coords, max_val, LUT=LUT)
            self.tuples.append(copy_tuple)

            for _ in range(3):
                rotated_coords = rotate_clockwise(rotated_coords)
                reflected_rotated_coords = rotate_clockwise(reflected_rotated_coords)

                self.tuples.append(NTuple(rotated_coords, max_val,LUT=LUT))
                self.tuples.append(NTuple(reflected_rotated_coords, max_val,LUT=LUT))


    def evaluate(self, board):
        res = 0
        for n_tuple in self.tuples:
            res += n_tuple.evaluate(board)
        return res
    
    def update(self, board, difference):
        for n_tuple in self.tuples:
            n_tuple.update(board, difference / len(self.tuples))



class TDAgent:
    def __init__(self, all_coords, max_val, learning_rate, trace_decay):
        NTN = NTupleNetwork(all_coords, max_val)
        self.NTN = NTN
        self.m = len(NTN.tuples)
        self.max_val = max_val

        self.history = []  # reset after each episode

        # TODO learning rate should appear somewhere
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay  # TD(lambda)
        if trace_decay == 0:
            self.h = 1
        else:
            self.h = int(math.log(0.1, trace_decay))  # cutoff index for updating history. earlier states have a weight of < trace_decay^h = 0.1

    def evaluate(self, state, move):
        """
            Evaluate an (state, action) pair using the current N-Tuple network
        
           Parameters:
           state ([[int]]): Board corresponding to the state
           move (int): Candidate move
        """
        afterstate, reward, _, info = Env2048._step(state, move, generate=False)
        v = self.NTN.evaluate(afterstate)
        return v + reward, info['valid_move']
    
    def update(self, reward, lost, won):
        if len(self.history) <= 1:
            return
        afterstate1 = self.history[-1]
        afterstate0 = self.history[-2]
        diff = reward + self.NTN.evaluate(afterstate1) - self.NTN.evaluate(afterstate0)
        if lost:
            diff -= self.NTN.evaluate(afterstate1)
        if won:
            diff = 10*(MAX_VAL**2)  # arbitrarily picked

        for k in range(1,self.h+1):
            t = len(self.history) - k
            if t < 0: 
                return
            afterstate = self.history[t]
            # changed_coords = set([(i,j) for i in range(4) for j in range(4) 
            #               if afterstate[i][j] != afterstate1[i][j]])
            ## doesn't really make sense to only limit updating to changed coordinates now 

            decay = 1 if self.trace_decay == 0 else (self.trace_decay**k)
            self.NTN.update(afterstate, self.learning_rate * diff * decay)


    # TODO some exploration?
    def move_greedy(self, state):
        evaluations = {}
        for move in range(4):
            val, legal = self.evaluate(state, move)
            if legal:
                evaluations[move] = val
        move, _ = max(list(evaluations.items()), key=lambda pair: pair[1])
        return move
    
    def save(self, folder_path='agents/', name=None):
        # TODO this will produce an unintuitive order for the agents when one deletes agent1 and keeps agent2
        base_filename = 'agent'
        extension = '.pkl'
        if name:
            path = folder_path + name + extension
        else: 
            i = 1
            while os.path.exists(os.path.join(folder_path, f'{base_filename}{i}{extension}')):
                i += 1

            path = os.path.join(folder_path, f'{base_filename}{i}{extension}')

        with open(path, 'wb') as file:
            pickle.dump(self.NTN, file)
        print(f'Agent saved in {path}')
    
    @classmethod
    def load(params, path):
        with open(path, 'rb') as file:
            NTN = pickle.load(file)
        agent = TDAgent([], 1) # TODO fix the initialization. [],1 are temporary values that are only used to initialize the NTN, which is instantly overwritten by the one we load
        agent.NTN = NTN
        print(f'Agent loaded from {path}')
        return agent
    
    def learn_from_episode(self, save=False):
        env = Env2048()
        state = env.reset()
        self.history = [state]  # list of afterstates
        done = False
        t = 0
        
        while not done:
            move = self.move_greedy(state)
            state, reward, lost, _ = env.step(move)
            won = np.max(state) >= 2**self.max_val
            self.update(reward, lost, won)

            self.history.append(state)
            t += 1

            done = lost or won
        

        if save:
            save_game(self.history, base_name='td')
        
        # TEMP
        # if env.score > 3000:
        #     save_game(history, base_name='td')

        return env.score, np.max(state)


# TODO copy what is in the original paper exactly
all_locations = [\
    [[0,0,0,0],
    [0,0,0,0],
    [1,1,0,0],
    [1,1,1,1]],

    [[0,0,0,0],
    [1,1,0,0],
    [1,1,1,1],
    [0,0,0,0]],

    [[1,1,0,0], 
     [1,1,1,1],
     [0,0,0,0],
     [0,0,0,0]],

    [[0,1,1,1], 
     [0,1,1,1],
     [0,0,0,0],
     [0,0,0,0]],

    [[0,0,0,0], 
     [0,1,1,1],
     [0,1,1,1],
     [0,0,0,0]],

     [[0,0,0,0], 
     [0,0,0,0],
     [0,1,1,1],
     [0,1,1,1]]
]
all_locations = [np.array(a) for a in all_locations]
all_coords = [np.argwhere(locations == 1) for locations in all_locations]
all_coords = [[tuple(row) for row in coords] for coords in all_coords]


# agent = TDAgent.load('agents/agent3.pkl')

learning_rate = 0.05
trace_decay = 0
agent = TDAgent(all_coords, MAX_VAL, learning_rate, trace_decay)

episode = 0
scores = []
top_tiles = []
update_freq = 20
while True:
    score, top_tile = agent.learn_from_episode()
    
    episode += 1
    scores.append(score)
    top_tiles.append(top_tile)

    if episode % update_freq == 0:
        avg_score = np.mean(scores[episode-update_freq:])
        avg_top_tile = np.mean(top_tiles[episode-update_freq:])
        percentage_perfect = np.count_nonzero(np.array(top_tiles[episode-update_freq:]) == 2**MAX_VAL) / update_freq
        print(f'Episode {episode}: average score {round(avg_score)}, average highest tile {round(avg_top_tile)}, fraction of perfect {percentage_perfect:.2f}')

        LUTs = [n_tuple.LUT for n_tuple in agent.NTN.tuples]  # contains duplicates but doesn't matter

        n_untouched = 0
        for LUT in LUTs:
            n_untouched += np.count_nonzero(LUT==OPTIMISTIC_INIT)
        frac_untouched = n_untouched / (len(LUTs) * len(LUTs[0]))
        print(f'Fraction of LUT elements untouched: {frac_untouched:.5f}')
        # print(f'Number of LUT elements untouched: {n_untouched}')

agent.save()


plt.plot(range(len(top_tiles)), top_tiles)


def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

window_size = 50

# Calculate the moving average
moving_avg = moving_average(scores, window_size)

plt.plot(moving_avg)
