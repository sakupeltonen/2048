import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
import math
import time
import json
import sys
from collections import deque
from environment import Env2048
from tools import save_game, save_arr, moving_average, generate_all_boards

# np.random.seed(2)

temp = [2**i for i in range(14)]
log2 = {temp[i]: i for i in range(1,14)}
log2[0] = 0



# TODO : 
# - this agent assumes environment doesn't modify the state in place! this is not true in this branch
# - also check that this agent doesn't modify the board
# - afterstates

class NTuple:
    def __init__(self, coords, max_val, initial_values=0, LUT=None, E=None, A=None):
        """
           Parameters:
           coords ([int]): list of board coordinates (order is important)
           - n=len(indices) as in NTuple
           max_val (type): Maximum possible value in array
           - max_val^n is the size of the lookup table
           LUT, E, A: pointers to arrays for other NTuples, when initialized as a symmetric copy
        """
        self.coords = coords
        n = len(coords)
        self.n = n
        self.max_val = max_val
        self.max_tile = 2**max_val
        size = max_val ** n
        self.size = size
        self.winning_reward = (2**max_val)  # TODO hardcoded

        if LUT is None:
            self.LUT = np.zeros(size) + initial_values
            self.E = np.zeros(size)
            self.A = np.zeros(size)
        else:
            self.LUT = LUT
            self.E = E
            self.A = A


    def index(self, board):
        """ Get index i in LUT corresponding to the tuple given by board[self.coords] 
        TODO explain how boards with max_tile are handled"""
        values = [board[i][j] for (i,j) in self.coords]
        values = [log2[v] for v in values]
        res = 0
        for i in range(self.n):
            res += values[i] * ((self.max_val-1) ** i)
        return res

    def evaluate(self, board):
        """ NTuple(board) returns the value of the NTuple for the given tuple at board[self.coords] """
        values = [board[i][j] for (i,j) in self.coords]
        if self.max_tile in values:
            return self.winning_reward
        return self.LUT[self.index(board)]

    def update(self, board, difference, diff_lambda):
        # TODO rename difference, diff_lambda to sth more informative
        i = self.index(board)
        if self.A[i] != 0:
            learning_rate = abs(self.E[i]) / self.A[i]
        else:
            learning_rate = 1

        self.LUT[i] += learning_rate * difference
        self.E[i] += diff_lambda
        self.A[i] += abs(diff_lambda)
        


class NTupleNetwork:
    def __init__(self, specs):
        self.tuples = [] 
        self.specs = specs
        self.init_tuples(specs)
        

    def init_tuples(self, specs):
        def reflect_horizontal(coords):
            return [(height-1-y,x) for (y,x) in coords]
        
        def rotate_clockwise(coords):
            # e.g. (y,x)=(3,3) --> (3,0); (y,x)=(2,1) --> (1,1) 
            return [(x,width-1-y) for (y,x) in coords]
        
        all_coords = NTupleNetwork.tuple_coords_from_layout(specs['layout'])
        height = specs['height']
        width = specs['width']
        max_val = log2[specs['max_tile']]
        symmetries = specs['symmetries']
        optimistic_init = specs['optimistic_init']

        assert (height == width) or (not symmetries), "Non-rectangular board doesn't support symmetries"

        for coords in all_coords:
            rotated_coords = [(y,x) for (y,x) in coords]
            reflected_rotated_coords = reflect_horizontal(coords)

            original_n_tuple = NTuple(rotated_coords, max_val, initial_values=optimistic_init)
            LUT = original_n_tuple.LUT
            E = original_n_tuple.E
            A = original_n_tuple.A
            self.tuples.append(original_n_tuple)

            if symmetries:
                copy_tuple = NTuple(reflected_rotated_coords, max_val, 
                                    initial_values=optimistic_init, LUT=LUT, E=E, A=A)
                self.tuples.append(copy_tuple)

                for _ in range(3):
                    rotated_coords = rotate_clockwise(rotated_coords)
                    reflected_rotated_coords = rotate_clockwise(reflected_rotated_coords)

                    self.tuples.append(NTuple(rotated_coords, max_val, 
                                              initial_values=optimistic_init,LUT=LUT, E=E, A=A))
                    self.tuples.append(NTuple(reflected_rotated_coords, max_val,    initial_values=optimistic_init ,LUT=LUT, E=E, A=A))


    def evaluate(self, board):
        res = 0
        for n_tuple in self.tuples:
            res += n_tuple.evaluate(board)
        return res
    
    def update(self, board, diff):
        beta = self.specs['meta_learning_rate']
        for n_tuple in self.tuples:
            n_tuple.update(board, (beta/len(self.tuples)) * diff, diff)

    @staticmethod
    def tuple_coords_from_layout(layout):
        layouts = [np.array(a) for a in layout]
        all_coords = [np.argwhere(locations == 1) for locations in layouts]
        all_coords = [[tuple(row) for row in coords] for coords in all_coords]
        return all_coords



class TDAgent:
    def __init__(self, specs):
        self.specs = specs

        self.width = specs['width']
        self.height = specs['height']
        self.max_val = log2[specs['max_tile']]
        
        NTN = NTupleNetwork(specs)
        self.NTN = NTN
        self.m = len(NTN.tuples)

        self.learning_rate = specs['start_epsilon']

        # reset after each episode
        self.history = []  
        self.afterstates = [] 
        self.diffs = []

        trace_decay = specs['trace_decay']  # TD(lambda)
        self.trace_decay = trace_decay
        if trace_decay == 0:
            self.h = 0
        else:
            self.h = int(math.log(specs['cut_off_weight'], trace_decay))  # cutoff index for updating history. earlier states have a weight of < trace_decay^h = 0.1

        self.queue = deque()
        self.diff_sum = 0
            

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
        afterstate1 = self.afterstates[-1]
        afterstate0 = self.afterstates[-2]

        if lost and not won:
            diff = reward - self.NTN.evaluate(afterstate0)
        else:
            diff = reward + self.NTN.evaluate(afterstate1) - self.NTN.evaluate(afterstate0)
        self.diffs.append(diff)

        t = len(self.history)-1  # initial state has index 0

        # h is zero if lambda=0
        # V(s_t) + diff = V(S_t+1) + r_t. diff = V(S_t+1) + r_t - V(S_t)
        # with lambda!= 0, also update earlier states s_t-k with a weight of lambda^k, up to k=h
        # maintain the sum of weighted updates, diff_t + diff_{t+1} lambda + ... diff_{t+h} lambda^h for s_t

        self.diff_sum *= self.trace_decay
        self.diff_sum += self.diffs[t]  # same as diff

        # e.g. 1 > 0. remove diff added at previous time step.
        # h=1
        # t=1: add first diff[1]
        # this stays in the sum since 1  > 1 false
        # t=2: sum becomes lamdba diff[1] + diff[2]
        # 2 > 1 true
        # delta_last = diff[2-1-1]= diff[0] = 0
        if t > self.h:
            # e.g. h=0: remove the diff added in the last iteration 
            delta_last = self.diffs[t-self.h-1]
            self.diff_sum -= delta_last * (self.trace_decay ** (self.h+1))

        if not (lost or won):
            if t >= self.h:
                # update state s'_{t-h} with the sum
                afterstate = self.afterstates[t-self.h]
                self.NTN.update(afterstate, self.diff_sum)

        else:
            # update states s_t-k, which is s_{t-h},...,s_{t}
            for k in reversed(range(self.h+1)):
                if t-k < 0: 
                    continue
                afterstate = self.afterstates[t-k]
                self.NTN.update(afterstate, self.diff_sum)

                delta_last = self.diffs[t-k]
                self.diff_sum -= delta_last * (self.trace_decay ** k)  
                


    def move_greedy(self, state):
        evaluations = {}
        for move in range(4):
            val, legal = self.evaluate(state, move)
            if legal:
                evaluations[move] = val
        move, _ = max(list(evaluations.items()), key=lambda pair: pair[1])
        return move
    
    def move_random(self, state):
        legal_moves = []
        for move in range(4):
            _, _, _, info = Env2048._step(state, move, generate=False)
            if info['valid_move']:
                legal_moves.append(move)
        return random.choice(legal_moves)
    
    def move(self, state):
        if random.random() < self.learning_rate:
            return self.move_random(state)
        else:
            return self.move_greedy(state)
        
    def update_learning_rate(self):
        self.learning_rate = max(self.specs['min_epsilon'], self.learning_rate * self.specs['decay_factor'])

    
    def save_old(self, folder_path='agents/', name=None, verbose=True):
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
        if verbose:
            print(f'Agent saved in {path}')

    def save(self, name, verbose=True):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = name + '.pkl'
        path = os.path.join(script_dir,'agents', filename)

        with open(path, 'wb') as file:
            pickle.dump(self.NTN, file)
        if verbose:
            print(f'Agent saved in {path}')
    
    @classmethod
    def load(cls, specs):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = specs['name'] + '.pkl'
        path = os.path.join(script_dir,'agents', filename)

        with open(path, 'rb') as file:
            NTN = pickle.load(file)
        agent = TDAgent(specs)  # creates a temp NTN, E, A before the real ones are loaded
        agent.NTN = NTN
        print(f'Agent loaded from {path}')
        return agent
    
    def learn_from_episode(self):
        start_tile = self.specs['start_tile']
        if start_tile == 0:
            env = Env2048(width=self.width, height=self.height)
            state = env.reset()
        else:
            env = Env2048.initial_with_tile(start_tile, width=self.width, height=self.height)
            state = env.board
        self.afterstates = [env.board]  # this is an empty board before spawning any tiles
        self.history = [state]  # list of states
        self.diffs = [0]  # the 0 maintains consistent indexing with history, afterstates. doesn't correspond to any move
        self.diff_sum = 0
        done = False
        
        while not done:
            move = self.move(state)
            state, reward, lost, info = env.step(move)
            won = np.max(state) >= 2**self.max_val
            done = lost or won

            afterstate = info['afterstate']
            self.afterstates.append(afterstate)
            self.history.append(state)

            self.update(reward, lost, won)

        return env.score, np.max(state)


# =========================================
#         MAIN LOOP
# =========================================

def train(agent, agent_specs, n_episodes, saving_on=True):
    scores = []
    top_tiles = []
    update_freq = agent_specs['update_freq']
    save_freq = agent_specs['save_freq']
    start_time = time.time()
    for episode in range(n_episodes):
        score, top_tile = agent.learn_from_episode()
        
        episode += 1
        scores.append(score)
        # TODO take start tile into account
        top_tiles.append(top_tile)

        if episode % agent_specs['learning_rate_decay_freq'] == 0:
            agent.update_learning_rate()

        if episode % update_freq == 0:
            avg_score = np.mean(scores[episode-update_freq:])
            avg_top_tile = np.mean(top_tiles[episode-update_freq:])
            end_time = time.time()

            max_tile = agent_specs['max_tile']
            max_val = log2[max_tile]
            percentage_perfect = np.count_nonzero(np.array(top_tiles[episode-update_freq:]) == 2**max_val) / update_freq
            seconds_per_game = (end_time-start_time) / update_freq
            print(f'Episode {episode}: avg score {round(avg_score)}, avg top tile {round(avg_top_tile)}, percentage perfect {percentage_perfect:.2f}, avg time {seconds_per_game:.2f}')
            start_time = time.time()

        if episode % save_freq == 0 and saving_on:
            agent.save(name=agent_specs['name'], verbose=False)
            save_arr(scores, agent_specs['name'], 'score')
            save_arr(top_tiles, agent_specs['name'], 'toptile')

        
# =========================================
#         AGENT INITIALIZATION
# =========================================

# arguments: agent specs filename, load/new, n_episodes
DEBUG = False

if __name__ == "__main__" and not DEBUG:
    agent_name = sys.argv[1]
    method = sys.argv[2]
    n_episodes = int(sys.argv[3])
    #saving_on = bool(sys.argv[4])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    specs_file = agent_name + '.json'
    path = os.path.join(script_dir,'specs', specs_file)
    agent_specs = json.load(open(path, "r"))

    max_tile = agent_specs['max_tile']
    max_val = log2[max_tile]
    width = agent_specs['width']
    height = agent_specs['height']
    
    if method=='load': 
        agent = TDAgent.load(agent_specs)
    else:
        agent = TDAgent(agent_specs)

    
    train(agent, agent_specs, n_episodes, saving_on=True)


# agent_specs = json.load(open('specs/3x3.json', 'r'))
# agent = TDAgent(agent_specs)
# n_episodes = 10000
# train(agent, agent_specs, n_episodes, saving_on=True)


# =========================================
#               DEBUGGING
# =========================================

# window_size = 500
# plt.plot(moving_average(scores, window_size))
# plt.plot(moving_average(top_tiles, window_size))



# vals = [0]
# vals += [2**i for i in range(1, max_val + 1)]

# ntuple = agent.NTN.tuples[0]

# for board in generate_all_boards(width,height,vals):
#     if ntuple.evaluate(board) > 0.1: 
#         print(f'{board}\t{ntuple.evaluate(board):.2f}\n')