import math
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        data = file.readlines()[1:]

    data = [list(map(float, row.strip().split(','))) for row in data]

    res = 97  # Prime number to avoid artificial hotspots

    heatmap = np.ones((res, res * 2))

    for row in data:
        left_x = int(row[2] * res)
        left_y = int(row[3] * res)
        heatmap[left_y, left_x] += 1

        right_x = int(row[4] * res)
        right_y = int(row[5] * res)
        heatmap[right_y, right_x] += 1

    img = plt.imshow(
        heatmap,
        cmap='viridis',
        interpolation='nearest',
        norm='log',
    )

    plt.xticks([])
    plt.yticks([])

    file_name = os.path.basename(sys.argv[1]).split('.')[0]
    plt.savefig(f"{file_name}_heatmap.png", dpi=300, bbox_inches='tight')
