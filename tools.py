import os
import pickle
import numpy as np
from datetime import datetime

def save_game(history, actions):
    """
       Save a game history and the taken actions to a file
    
       Parameters:
       history ([board]): List of game boards
       actions ([int]): List of taken actions
    """
    height = len(history[0])
    width = len(history[0][0])

    now = datetime.now()
    timestamp = now.strftime('%d%b-%H-%M')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir,'games', f'{height}x{width}-{timestamp}.pkl')

    with open(path, 'wb') as file:
        pickle.dump({'history': history, 'actions': actions}, file)

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