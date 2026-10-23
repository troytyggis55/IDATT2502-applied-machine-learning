import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        data = np.array([list(map(int, line.split(","))) for line in file.read().splitlines()[1:]])

    # sort first by column 0, then by column 1
    data = data[np.lexsort((data[:, 1], data[:, 0]))]
    print(data)

    # Perfect test data
    #data = []
    #for i in range(1, 12):
    #    for j in range(1, 12):
    #        data.append([i * 1e6, j * 1e6, ((i - j) + 10) * 50])
    #data = np.array(data)

    # for unique values in column 0, plot a line with column 1 as x and column 2 as y
    # The older the model, the brighter the line, use cmap viridis
    def viridis_interpolation(x):
        return matplotlib.colors.Normalize(vmin=0, vmax=data[:, 0].max())(x)
    
    #for unique in np.unique(data[:, 0]):
    #    subset = data[data[:, 0] == unique]
    #    plt.plot(subset[:, 1], subset[:, 2], color=plt.cm.viridis(viridis_interpolation(unique)), alpha=0.3)

    for unique in np.unique(data[:, 0]):
        subset = data[data[:, 0] == unique]
        plt.scatter(subset[:, 1], subset[:, 2], color=plt.cm.viridis(viridis_interpolation(unique)),
                    alpha=1)
        
    # Plot the average
    for unique in np.unique(data[:, 1]):
        subset = data[data[:, 1] == unique]
        plt.plot([unique] * len(subset), subset[:, 2], color='black', alpha=1)

    # Add labels and title
    plt.xlabel("Millions of steps")
    plt.ylabel("Number of wins")
    plt.title("Wins after 1000 games")
    
    plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=data[:, 0].max()), cmap='viridis'), ax=plt.gca())

    plt.xticks(ticks=data[:, 1], labels=data[:, 1])
    formatter = FuncFormatter(lambda x, _: f'{(x/1e6):.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().tick_params(axis='both', which='major', labelsize=8)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(100))

    plt.grid(True)
    
    fig = plt.gcf()
    fig.set_size_inches(14, 10)

    # plt.show()

    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
    else:
        plt.savefig("GraphPlot.png", dpi=300, bbox_inches='tight')




