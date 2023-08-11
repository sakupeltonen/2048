import os
import pickle

def save_game(history, folder_path='games/', name=None, base_name = 'game'):
    """
       Parameters:
       history ([np.array]): List of board states
       name (string): Exact name of the file
       base_name (string): Prefix of filename. Doesn't overwrite, instead saved as base_name{x} where x is the first free integer
    """
    extension = '.pkl'
    if name:
        path = folder_path + name + extension
    else: 
        i = 1
        while os.path.exists(os.path.join(folder_path, f'{base_name}{i}{extension}')):
            i += 1
        path = os.path.join(folder_path, f'{base_name}{i}{extension}')

    with open(path, 'wb') as file:
        pickle.dump(history, file)