import os
import pickle

def save_game(history, name):
    """
       Parameters: TODO update
       history ([np.array]): List of board states
       name (string): Exact name of the file
       base_name (string): Prefix of filename. Doesn't overwrite, instead saved as base_name{x} where x is the first free integer
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = name + '.pkl'
    path = os.path.join(script_dir,'games', filename)

    with open(path, 'wb') as file:
        pickle.dump(history, file)