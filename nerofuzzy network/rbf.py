import numpy as np
import random as r
class Rbf:
    def __init__(self, xi_in, y_in, u, c, m, dia, clus):
        self.xi_in = xi_in  #shuffle
        self.y_in = y_in    #shuffle

        self.u = u  #shuffle
        self.c = c
        self.m = m
        self.dia = dia
        self.clus = clus

        self.xi_learn = []
        self.u_learn = []
        self.y_learn = []
        self.y0_learn = []

        self.xi_test = []
        self.u_test = []
        self.y_test = []
        self.y0_test = []


        self.g = []
        self.w = []
        self.yhat = []
        self.acc = 0.0

    def separate(self):
        c = list(zip(self.xi_in, self.y_in, self.u))
        np.random.shuffle(c)
        self.xi_in, self.y_in, self.u = zip(*c)
        sev = int(len(self.xi_in)*0.7)

        self.xi_learn = self.xi_in[:sev]
        self.y_learn = self.y_in[:sev]
        self.u_learn = self.u[:sev]

        self.xi_test = self.xi_in[sev:]
        self.y_test = self.y_in[sev:]
        self.u_test = self.u[sev:]


    def learn(self):
        self.g_calc(self.xi_learn, self.u_learn)
        self.w_calc(self.xi_learn, self.y_learn)
        self.yhat_calc()
        self.acc_calc(self.y0_learn)

    def test(self):
        self.g_calc(self.xi_test, self.u_test)
        self.yhat2_calc()
        self.acc_calc(self.y0_test)

    def g_calc(self, data, u):
        g_temp = []
        for i in range(self.clus):
            gi_temp = []
            for k in range(len(data)):
                xk_vi = np.subtract(data[k], self.c[i])
                # ci = self.co_calc(i, np.array(xk_vi)[np.newaxis], data, u)

                diamult = np.multiply(-self.dia, xk_vi)
                # inv = (1/ci).dot(np.array(xk_vi)[np.newaxis].transpose())
                # inv = np.linalg.inv(ci).dot(np.array(xk_vi)[np.newaxis].transpose())
                # inv = ci.dot(np.array(xk_vi)[np.newaxis].transpose())
                inv = np.array(xk_vi)[np.newaxis].transpose()
                dot = diamult.dot(inv)

                xk = np.exp(dot)
                gi_temp.append(xk[0])
            g_temp.append(gi_temp)
        self.g = np.asarray(g_temp).transpose()

    def co_calc(self, i, xk_vi, data, u):
        top = np.array([0, 0])
        but = 0

        for k in range(len(data)):
            # xk_vi.transpose().dot(xk_vi)
            top = np.add(top,
                         np.multiply(u[k][i] ** self.m, xk_vi.transpose().dot(xk_vi))
                         )
            but += u[k][i] ** self.m
        res = np.divide(top, but)
        ci_temp = np.asarray(res)
        # print('ci',np.add(ci_temp, 0.01))
        return np.add(ci_temp, 0.01)
        # return np.array([[r.random(),r.random()],[r.random(),r.random()]])

    def w_calc(self, xi, y):
        gt = self.g.transpose()
        y0 = np.subtract(np.array(y), 1)
        self.y0_learn = y0
        classes = len(np.unique(y))
        ylabel = np.zeros([len(xi), classes], dtype=int)
        ylabel[np.arange(len(y)), y0] = 1
        ret = gt.dot(self.g).dot(gt).dot(ylabel)
        self.w = ret

    def yhat_calc(self):
        res = self.g.dot(self.w)
        self.yhat = np.argmax(res, axis=1)
        # print("g",self.g)
        # print('w',self.w)
        # print('res',res)

    def yhat2_calc(self):
        res = self.g.dot(self.w)
        self.yhat = np.argmax(res, axis=1)
        y0 = np.subtract(np.array(self.y_test), 1)
        self.y0_test = y0
        # print("g",self.g)
        # print('w',self.w)
        # print('res',res)


    def acc_calc(self, y):
        print("---------------------------")
        # print(y)
        # print(self.yhat)
        self.acc = 1 - np.sum(np.abs(np.sign(np.subtract(self.yhat, y)))) / len(y)
        print('acc',self.acc)



