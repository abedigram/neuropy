import plot
import file_handler
import network
import learner
import numpy as np
import copy

n_epoch = 50
lr = 0.01

data = file_handler.read_csv('data')

# shuffled data and picked half
np.random.shuffle(data)
train_data = copy.deepcopy(data[:(len(data)//2)])
test_data = copy.deepcopy(data[(len(data)//2):])
print(len(train_data))

# initialize network
net0 = network.Network([2, 1], n_epoch=n_epoch, lr=lr)
# net1 = network.Network([2, 1], n_epoch=3000, lr=3)
# learn network
learner.learn(net0, train_data)
res = net0.test(test_data)
# plot.plot(data,'input', 0.5)
plot.plot(res,'res', net0.bs[0][0][0],net0)

net1 = network.Network([2,1], n_epoch=n_epoch, lr=lr)
net2 = network.Network([2,1], n_epoch=n_epoch, lr=lr)
net3 = network.Network([2,1], n_epoch=n_epoch, lr=lr)

learner.learn2(net1,net2,net3,data)
res2 = learner.test2(net1,net2,net3,test_data)
plot.plot(res2,'res2', net3.bs[0][0][0]/2,net2)














