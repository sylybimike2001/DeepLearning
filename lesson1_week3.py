import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
decrease_rate = 0.01

def ReLU(X):
    return max(0,X)

def print_test(shape_X,shape_Y):
    print("X的大小："+str(shape_X))
    print("Y的大小:"+str(shape_Y))
    print("数据规模:"+str(shape_Y[1]))

def layer_sizes(X,Y):
    n_x = X.shape[0]    #输入层节点数
    n_h = 4             #隐藏层节点数
    n_y = Y.shape[0]    #输出层节点数

    return n_x,n_h,n_y

def initialize_parameters( n_x , n_h ,n_y):
    #np.random.seed(2)

    W1 = np.random.randn(n_h,n_x) * decrease_rate
    B1 = np.random.randn(n_h,1)
    W2 = np.random.randn(n_y,n_h) * decrease_rate
    B2 = np.random.randn(n_y,1)

    parameters = {"W1": W1,
                  "b1": B1,
                  "W2": W2,
                  "b2": B2}

    return parameters

def ReLuFunc(x):
    # ReLu 函数
    x = (np.abs(x) + x) / 2.0
    return x


def ReLuPrime(x):
    # ReLu 导数
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def forward_propagation( X , parameters ,status = "tanh"):
    """

    :param X: 输入数据集
    :param parameters: W1,B1,W2,B2
    :return: ..
    """\
    #接受参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #进行计算,向前传播
    Z1 = np.dot(W1,X)+b1
    if(status == "sigmoid"):A1 = sigmoid(Z1)

    if(status == "tanh"):A1 = np.tanh(Z1)

    if(status == "ReLU"):A1 = ReLuFunc(Z1)

    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)

def compute_cost(A2,Y,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    m = Y.shape[1]

    temp = np.multiply(Y,np.log(A2))+np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(temp)/m
    cost = float(np.squeeze(cost))
    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters,cache,X,Y,status = "tanh"):
    m = Y.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y  #dZ2是构建dW2和dB2的基础
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    dB2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)

    if(status == "tanh"):dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))

    if(status == "ReLU"):dZ1 = np.multiply(np.dot(W2.T,dZ2),ReLuPrime(Z1))

    dW1 = (1/m)*np.dot(dZ1,X.T)

    dB1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": dB1,
             "dW2": dW2,
             "db2": dB2}

    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X,Y,n_h,num_iterations=10000,print_cost = False):
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if print_cost:
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))
    return parameters

def predict(parameters,X):

    A2,cache = forward_propagation(X,parameters)

    return np.round(A2)

X, Y = load_planar_dataset()
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # 训练集里面的数量

parameters = nn_model(X, Y, n_h = 4, num_iterations=30000, print_cost=True)


predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')





