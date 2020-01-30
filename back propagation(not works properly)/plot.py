import matplotlib.pyplot as plt
import numpy as np
def plot(data,name, b,net):
    print(b)
    # plot results
    zerox = []
    zeroy = []
    onex = []
    oney = []
    for i in data:
        print(i[2])
        if i[2] < b:
            zerox.append(i[0])
            zeroy.append(i[1])
        else:
            onex.append(i[0])
            oney.append(i[1])
    # print(net.ws)
    plt.scatter(zerox, zeroy, s=10, color='b')
    plt.scatter(onex, oney, s=10, color='r')
    plt.title('reds are 1 - blues are 0')
    # plt.plot(net.ws[0][0]+net.bs[0][0])
    # plt.plot(net.ws[0][0])
    plt.savefig('.//'+name+'.png')
    plt.show()




