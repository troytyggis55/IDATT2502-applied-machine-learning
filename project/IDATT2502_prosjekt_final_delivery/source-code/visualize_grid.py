import math
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

    min_steps = data[:, 0].min()
    max_steps = data[:, 0].max()

    size = int(np.sqrt(len(data)))
    formatted = np.array(data)[:, 2].reshape(size, size)

    img = plt.imshow(
        formatted,
        cmap='viridis',
        interpolation='nearest',
        norm=matplotlib.colors.Normalize(vmin=0, vmax=data[:, 2].max()),
        extent=[min_steps, max_steps, max_steps, min_steps],
        aspect='equal'
    )
    
    plt.xticks(ticks=data[:, 0], labels=data[:, 0])
    plt.yticks(ticks=data[:, 0], labels=data[:, 0])

    # Add labels and title
    plt.xlabel("Right model trainingsteps")
    plt.ylabel("Left model trainingsteps")
    plt.title("Left wins against right")

    plt.colorbar(img)

    fig = plt.gcf()
    fig.set_size_inches(13, 10)

    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"GridPlot_{datetime.now().strftime('%d%m_%H%M')}.png", dpi=300, bbox_inches='tight')
