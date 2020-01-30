import numpy as np
import copy
import math
import matplotlib.pyplot as plt

class Network:

    def __init__(self, layers, n_epoch, lr):
        self.n_epoch = n_epoch
        self.lr = lr

        self.layers = layers
        self.bs = [np.random.randn(y, 1) for y in layers[1:]]
        self.ws = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

        self.grad_ws = copy.deepcopy(self.ws)
        self.grad_bs = copy.deepcopy(self.bs)
        # print(self.bs)
        # print(self.ws)

    def sigmoid(self,z):
        return 1.0 / (1.0 + math.exp(-z))

    def compute_grad(self, data):

        self.set_grad_0_w()
        for i in range(len(self.grad_ws)):
            for datum in data:  # datum[x0,x1,label]
                s_in = np.dot(self.ws, datum[:-1])[0] + self.bs[0]
                y = self.sigmoid(s_in)
                self.grad_ws[i] += abs(y-datum[-1])*datum[i]*y*(1-y)
        for i in range(len(self.ws[0])):
            self.ws[i] -= (self.lr * self.grad_ws[0])

        self.set_grad_0_b()
        for datum in data:  # datum[x0,x1,label]
            s_in = np.dot(self.ws, datum[:-1])[0] + self.bs[0]
            y = self.sigmoid(s_in)
            self.grad_bs[0] += abs(y - datum[-1]) * y * (1 - y)
        self.bs[0] -= (self.lr * self.grad_bs[0])

    def set_grad_0(self, el):
        for i in range(len(el)):
            for j in range(len(el[i])):
                for k in range(len(el[i][j])):
                    el[i][j][k] = 0

    def set_grad_0_w(self):
        self.set_grad_0(self.grad_ws)

    def set_grad_0_b(self):
        self.set_grad_0(self.grad_bs)

    def test(self, data):
        for i in range(len(data)):
            data[i][2] = self.sigmoid(np.dot(self.ws, data[i][:-1])[0]+self.bs[0])
        return data

    def test1(self, data):
        ys = []
        for i in range(len(data)):
            ys.append(self.sigmoid(np.dot(self.ws, data[i][:-1])[0] + self.bs[0]))
        return ys

    def compute_grad_mid(self, data):
        self.set_grad_0_w()
        self.set_grad_0_b()
        ys = []
        for i in range(len(self.grad_ws)):
            for datum in data:  # datum[x0,x1,label]
                s_in = np.dot(self.ws, datum[:-1])[0] + self.bs[0]
                y = self.sigmoid(s_in)
                ys.append(y)
        return ys;

    def grad(self,ys,data):
        for i in range(len(self.grad_ws)):
            for datum , y in zip(data,ys):  # datum[x0,x1,label]
                self.grad_ws[i] += abs(y-datum[-1])*datum[i]*y*(1-y)
        for i in range(len(self.ws[0])):
            self.ws[i] -= (self.lr * self.grad_ws[0])

        for datum in data:  # datum[x0,x1,label]
            s_in = np.dot(self.ws, datum[:-1])[0] + self.bs[0]
            y = self.sigmoid(s_in)
            self.grad_bs[0] += abs(y - datum[-1]) * y * (1 - y)
        self.bs[0] -= (self.lr * self.grad_bs[0])


