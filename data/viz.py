import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm



def visualize_and_compare_poses(sequence_idx):
    old_df = pd.read_csv(f'/home/minh/converted/{sequence_idx}/poses_full.txt', sep=' ', names=[str(i) for i in range(12)])
    x_old = np.array(old_df['3'])
    y_old = np.array(old_df['7'])

    new_df = pd.read_csv(f'/home/minh/converted/{sequence_idx}/poses.txt', sep=' ', names=[str(i) for i in range(12)])
    x_new = np.array(new_df['3'])
    y_new = np.array(new_df['7'])

    print(len(x_old))
    print(len(x_new))

    plt.figure(0)
    plt.plot(x_old, y_old, 'bo')
    plt.plot(x_new, y_new, 'rx')
    plt.show()
            

visualize_and_compare_poses(2)