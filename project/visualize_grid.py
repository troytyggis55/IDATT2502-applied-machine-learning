import math
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        data = np.array([list(map(int, line.split(","))) for line in file.read().splitlines()[1:]])

    #sort first by column 0, then by column 1
    data = data[np.lexsort((data[:, 1], data[:, 0]))]    
    print(data)
    
    #Perfect test data
    #data = []
    #for i in range(10):
    #    for j in range(10):
    #        data.append([i*1e6, j*1e6, ((i-j)+10)*50+50])
    #        
    #data = np.array(data)
    
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

    formatter = FuncFormatter(lambda x, _: f'{(x/1e6):.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().tick_params(axis='both', which='major', labelsize=8)

    # Add labels and title
    plt.xlabel("Millions of trainingsteps")
    plt.ylabel("Millions of trainingsteps")
    plt.title("Wins after 1000 games")
    
    plt.colorbar(img)

    fig = plt.gcf()
    fig.set_size_inches(13, 10)
    
    #plt.show()
    
    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
    else:
        plt.savefig("GridPlot.png", dpi=300, bbox_inches='tight')
    
