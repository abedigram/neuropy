

def learn(net, data):
    # for i in range(1):
    for i in range(net.n_epoch):
        print(i)
        net.compute_grad(data)
        # net.test(test)
        print(i+1)


def learn2(net1, net2, net3, data):
    for i in range(net1.n_epoch):
        print(i)
        x0 = net1.compute_grad_mid(data)
        x1 = net2.compute_grad_mid(data)
        dat = []
        for i,j,k in zip(x0,x1,data):
            dat.append([i,j,k[2]])
        y = net3.compute_grad_mid(dat)
        net1.grad(y,data)
        net2.grad(y,data)
        net3.grad(y,data)


def test2(net1,net2,net3,data):

    res1 = net1.test1(data)
    res2 = net2.test1(data)
    dat = []
    for i, j in zip(res1, res2):
        dat.append([i,j,0])
    res3 = net3.test1(dat)

    for i in range(len(data)):
        data[i][2] = res3[i]

    return data