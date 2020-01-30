import file_reader
import fcm
import rbf
import plot
import copy
import numpy as np
file = 'data/12004.csv'
clus = 4
dia = 0.7

m = 2


def labeling(data, label):
    if not isinstance(label, list):
        res_data = []
        data = np.asarray(data)
        for i in range(len(data)):
            res_data.append(np.append(data[i, :], label).tolist())
        return np.asarray(res_data)
    else:
        res_data = []
        data = np.asarray(data)
        for i in range(len(data)):
            res_data.append(np.append(data[i, :], label[i]).tolist())
        return np.asarray(res_data)

def result():
    red = []
    green = []
    res = []

    xi = np.array(copy.deepcopy(r.xi_test))

    y = np.subtract(copy.deepcopy(r.y_test), 1)
    yh = copy.deepcopy(r.yhat)
    for i in range(len(xi)):
        if y[i] == yh[i]:
            green.append([xi[i][0], xi[i][1], 0])
        else:
            red.append([xi[i][0], xi[i][1], 1])

    print()
    if red:
        res = np.append(green, red,  axis=0)
    else:
        return green
    return res


xi_in, y_in, in_whole = file_reader.reader(file)

f = fcm.Fcm(xi_in=xi_in, clus=clus, m=m)
f.optimize()

cc = np.append(in_whole, labeling(f.c, 8),  axis=0)

plot.plot_input_data(cc, 'center_mode')


r = rbf.Rbf(xi_in=xi_in, y_in=y_in, u=f.u, c=f.c, m=m, dia=dia, clus=clus)
r.separate()
r.learn()
r.test()

e = result()
plot.plot_input_data(result(), 'result_mode')

