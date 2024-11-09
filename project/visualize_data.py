import sys
import matplotlib
import matplotlib.pyplot as plt

def visualize_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split('\t') for line in file]
    x, y = zip(*[(int(row[0]), float(row[1])) for row in data])
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('File Data Visualization')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        visualize_file(sys.argv[1])