import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        data = np.array([list(map(int, line.split(","))) for line in file.read().splitlines()[1:]])

    data = data[np.lexsort((data[:, 1], data[:, 0]))]
    print(data)

    def viridis_interpolation(x):
        return matplotlib.colors.Normalize(vmin=0, vmax=data[:, 0].max())(x)
    
    for unique in np.unique(data[:, 0]):
        subset = data[data[:, 0] == unique]
        plt.plot(subset[:, 1], subset[:, 2], color=plt.cm.viridis(viridis_interpolation(unique)))

    # Add labels and title
    plt.xlabel("Right model trainingsteps")
    plt.ylabel("Number of wins")
    plt.title("Left wins against right")
    
    plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=data[:, 0].max()), cmap='viridis'), ax=plt.gca())

    plt.xticks(ticks=data[:, 1], labels=data[:, 1])
    
    plt.grid(True)
    
    fig = plt.gcf()
    fig.set_size_inches(14, 10)

    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"LinePlot_{datetime.now().strftime('%d%m_%H%M')}.png", dpi=300, bbox_inches='tight')




