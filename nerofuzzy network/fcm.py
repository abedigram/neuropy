import random
import numpy as np
import copy


class Fcm:
    def __init__(self, xi_in, clus, m):
        self.xi_in = xi_in
        self.clus = clus
        self.m = m

        self.u = []
        self.c = np.array([])
        # self.u = np.full((self.xi_in.shape[0], clus), 1/self.clus)
        # self.u = np.random.uniform(0, 1, size=(self.xi_in.shape[0], clus))

        self.u = []
        for i in range(len(xi_in)):
            rand_placeholders = []
            for j in range(clus - 1):
                rand_placeholders.append(random.random())
            rand_placeholders = np.sort(rand_placeholders)
            ui = []
            for j in range(clus):
                if j == 0:
                    ui.append(rand_placeholders[0])
                elif j == clus - 1:
                    ui.append(1 - rand_placeholders[j - 1])
                else:
                    ui.append(rand_placeholders[j] - rand_placeholders[j - 1])
            self.u.append(ui)
        self.u = np.asarray(self.u)

    def c_calc(self):
        self.c = []
        for j in range(self.clus):
            top = np.array([0, 0])
            but = 0
            for i in range(len(self.xi_in)):
                top = np.add(top ,self.xi_in[i].dot(self.u[i][j]**self.m))
                but += self.u[i][j]**self.m
            res = np.divide(top, but)
            self.c.append(res.tolist())

        self.c = np.asarray(self.c)

    def u_calc(self):
        for i in range(len(self.u)):
            for j in range(len(self.u[i])):
                but = 0
                for k in range(len(self.c)):
                    but += (np.linalg.norm(np.subtract(self.xi_in[i], self.c[j]))/\
                           np.linalg.norm(np.subtract(self.xi_in[i], self.c[k])))**\
                           (2/(self.m-1))
                self.u[i][j] = 1/but

    def optimize(self):
        for i in range(100):
            temp = copy.deepcopy(self.u)

            self.c_calc()
            self.u_calc()

            if self.test(temp) < 0.1:
                break

    # def printc(self,p):
    #     for x in p:
    #         print(*x, sep=" ")

    def test(self,temp):
        m = np.amax(np.absolute(np.subtract(temp, self.u)))
        print(m)
        return m
