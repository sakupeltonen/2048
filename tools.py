import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

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


def visualize_qval(net, width, height, title=None, cuda=False):
    """Plot Q-values in a grid shaped like the game board"""

    fig, axs = plt.subplots(height, width, figsize=(10, 10))
    
    for i in range(height):
        for j in range(width):
            # Get the index for the input neuron
            neuron_index = i*width + j
            # Extract the weights for the 4 outputs corresponding to the current input neuron
            if cuda:
                weights = net.weight[:, neuron_index].detach().cpu().numpy()
            else:
                weights = net.weight[:, neuron_index].detach().numpy()
            
            # Plot the weights as numbers near the four sides of each cell
            axs[i, j].text(0.5, 1.1, f"{weights[3]:.2f}", ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].text(0.5, -0.1, f"{weights[1]:.2f}", ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].text(-0.1, 0.5, f"{weights[0]:.2f}", ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].text(1.1, 0.5, f"{weights[2]:.2f}", ha='center', va='center', transform=axs[i, j].transAxes)

        #     pg.K_LEFT: 0, 
        # pg.K_RIGHT: 2,
        # pg.K_UP: 3,
        # pg.K_DOWN: 1

            axs[i, j].set_xticks([0, 1])
            axs[i, j].set_yticks([0, 1])
            axs[i, j].grid(True)

            # Turn off the ticks and labels while leaving the grid visible
            axs[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    plt.tight_layout()

    if title:
        fig.suptitle(title)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(script_dir + '/plots/' + title + ".jpg", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    
    
