import matplotlib.pyplot as plt
import numpy as np


def plot_input_data(input_data, mode):
    plt_data = []
    colors_first_mode = ['red', 'blue', 'yellow', 'green', 'orange', 'black', 'cyan', 'magenta', 'purple', 'brown', 'pink', 'gray', 'olive', 'aqua', 'azure', 'beige', 'coral', 'darkblue', 'gold', 'lavender', 'lightgreen']
    colors_second_mode = ['green', 'red']
    for i in range(len(input_data)):
        isFind = False
        for j in range(len(plt_data)):
            if plt_data[j][0] == float(input_data[i][2]):
                subType = [float(input_data[i][0]), float(input_data[i][1])]
                plt_data[j][1].append(subType)
                isFind = True
                break
        if not isFind:
            subType = [[float(input_data[i][0]), float(input_data[i][1])]]
            plt_data.append((float(input_data[i][2]), subType))
    datas = []
    for i in range(len(plt_data)):
        datas.append(np.asarray(plt_data[i][1]))
    colors = []
    if mode == 'result_mode':
        colors = colors_second_mode
    else:
        colors = colors_first_mode[:len(plt_data)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data, color in zip(datas, colors):
        x0 = data[:, 0]
        x1 = data[:, 1]
        ax.scatter(x0, x1, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

