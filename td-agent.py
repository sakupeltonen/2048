import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from environment import Env2048
from tools import save_game

np.random.seed(42)

temp = [2**i for i in range(12)]
log2 = {temp[i]: i for i in range(1,12)}
log2[0] = 0

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
        size = max_val ** n
        self.size = size

        if LUT is None:
            self.LUT = np.zeros(size)
            # TODO optimistic initialization to encourage exploration
        else:
            self.LUT = LUT

        if update_counts is None:
            self.update_counts = np.zeros(size) + 25000
        else:
            self.update_counts = update_counts

    def index(self, board):
        """ Get index i in LUT corresponding to the tuple given by board[self.coords] """
        # TODO the board contains values in the original format whereas this uses the powers
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

    def index_inverse(self, i):
        """Inverse operation of index. Used for debugging"""
        # TODO
        _i =  i
        values = [0 for _ in range(self.n)]
        for i in reversed(range(self.n)):
            # Find largest value at index i,
            # such that values[i] * (self.max_val ** i) <= i
            value = 0
            while True:
                if (value + 1) * (self.max_val ** i) <= _i:
                    value += 1
                else:
                    break
            values[i] = value
            _i -= value * (self.max_val ** i)

        check = self.index(values)
        assert check == i, ""
        return values


class NTupleNetwork:
    def __init__(self, all_coords, max_val):
        self.tuples = [NTuple(coords, max_val)
                         for coords in all_coords]
        # TODO add the 8 symmetries

    def evaluate(self, board):
        res = 0
        for n_tuple in self.tuples:
            res += n_tuple.evaluate(board)
        return res

    def update(self, board, difference, changed_indices):
        changed_tuples = [n_tuple for n_tuple in self.tuples 
                          if len(changed_indices.intersection(set(n_tuple.coords))) > 0]

        for n_tuple in changed_tuples:
            n_tuple.update(board, difference / len(changed_tuples))



class TDAgent:
    def __init__(self, all_coords, max_val):
        NTN = NTupleNetwork(all_coords, max_val)
        self.NTN = NTN
        self.m = len(NTN.tuples)
        # TODO learning rate should appear somewhere
        self.learning_rate = 0.1

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
    
    def update(self, afterstate0, afterstate1, reward, done=False):
        # TODO also update earlier states, stored in some history for the episode
        if not done:
            diff = reward + self.NTN.evaluate(afterstate1) - self.NTN.evaluate(afterstate0)
        else:
            diff = reward - self.NTN.evaluate(afterstate0)
        # TODO remove hardcoding of board width and height
        changed_coords = set([(i,j) for i in range(4) for j in range(4) 
                          if afterstate0[i][j] != afterstate1[i][j]])
        self.NTN.update(afterstate0, self.learning_rate * diff, changed_coords)

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
    
    @classmethod
    def load(params, path):
        with open(path, 'rb') as file:
            NTN = pickle.load(file)
        agent = TDAgent([], 1) # TODO fix the initialization. [],1 are temporary values that are only used to initialize the NTN, which is instantly overwritten by the one we load
        agent.NTN = NTN
        return agent
    
    def learn_from_episode(self, save=False):
        env = Env2048()
        state = env.reset()
        history = [state]  # list of afterstates
        done = False
        t = 0
        afterstate0 = None
        afterstate1 = None
        while not done:
            move = self.move_greedy(state)
            state, reward, done, info = env.step(move)
            afterstate1 = info['afterstate']

            if t > 0: 
                self.update(afterstate0, afterstate1, reward)

            history.append(state)
            afterstate0 = afterstate1
            t += 1
        
        self.update(afterstate0, afterstate1, reward, done=True)
        # TODO add ending condition for reaching 2024

        if save:
            save_game(history, base_name='td')
        
        # TEMP
        if env.score > 3000:
            save_game(history, base_name='td')

        return env.score, np.max(afterstate1)


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

# agent = TDAgent(all_coords, 11)
agent = TDAgent.load('agents/agent1.pkl')

episode = 0
scores = []
top_tiles = []
while episode < 100:
    score, top_tile = agent.learn_from_episode()
    print(f'Episode {episode}: score {score}, highest tile {top_tile}')
    episode += 1
    scores.append(score)
    top_tiles.append(top_tile)

agent.save()




