import os
import pickle
import numpy as np

def save_game(history, name):
    """
       Save a game history to a file
    
       Parameters:
       history ([board]): List of game boards
       name (string): File name
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = name + '.pkl'
    path = os.path.join(script_dir,'games', filename)

    with open(path, 'wb') as file:
        pickle.dump(history, file)

def save_arr(arr, agent_name, arr_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = agent_name + '-' + arr_name + '.txt'
    path = os.path.join(script_dir,'logs', filename)

    with open(path, 'w') as file:
        for item in arr:
            file.write("%s\n" % item)


def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def generate_all_boards(w, h, vals):
        
    def _generate_all(prefix):
        res = []
        for val in vals:
            a = [x for x in prefix]
            a.append(val)
            if len(a) == w * h:
                res.append(a)
            else:
                res += _generate_all(a)
        return res
    
    lists = _generate_all([])
    return [np.array(l).reshape((h,w)) for l in lists]