import math
import time
import numpy as np
from client import ClientA,ClientB,ClientC
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics  import accuracy_score, f1_score


def load_data():
    """加载数据"""
    breast = load_breast_cancer()
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, random_state=1)
    # 数据标准化
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    return X_train, y_train, X_test, y_test


## 将特征分配给A和B
def vertically_partition_data(X, X_test, A_idx, B_idx):
    """
    Vertically partition feature for party A and B
    :param X: train feature
    :param X_test: test feature
    :param A_idx: feature index of party A
    :param B_idx: feature index of party B
    :return: train data for A, B; test data for A, B
    """
    XA = X[:, A_idx]  
    XB = X[:, B_idx]  
    XB = np.c_[np.ones(X.shape[0]), XB]
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]
    XB_test = np.c_[np.ones(XB_test.shape[0]), XB_test]
    
    return XA, XB, XA_test, XB_test

def scaled_sigmoid(x):
    """稳定的 Sigmoid 映射到 (-1,1)"""
    # x = np.clip(x, -50, 50)  # 防止数值溢出
    return 2 / (1 + np.exp(-x)) - 1

def vertical_logistic_regression(X, y, X_test, y_test, config):
    """
    Start the processes of the three clients: A, B and C.
    :param X: features of the training dataset
    :param y: labels of the training dataset
    :param X_test: features of the test dataset
    :param y_test: labels of the test dataset
    :param config: the config dict
    :return: True
    """
    
    ## 获取数据
    XA, XB, XA_test, XB_test = vertically_partition_data(X, X_test, config['A_idx'], config['B_idx'])   #把数据按照特征分开
    print('XA:',XA.shape, '   XB:',XB.shape)
    
    ## 各参与方的初始化
    client_A = ClientA(XA, y, config)
    print("Client_A successfully initialized.")
    client_B = ClientB(XB, config)
    print("Client_B successfully initialized.")
    client_C =  ClientC(XA.shape, XB.shape, config)
    print("Client_C successfully initialized.")
    
    ## 各参与方之间连接的建立
    client_A.connect("B", client_B)
    client_A.connect("C", client_C)
    client_B.connect("A", client_A)
    client_B.connect("C", client_C)
    client_C.connect("A", client_A)
    client_C.connect("B", client_B)
    
    ## 训练
    for i in range(config['n_iter']):
        client_C.task_1("A", "B")   #生成paillier的密钥对
        # stime = time.time()
        client_B.task_1("A")
        # etime = time.time()
        # t += etime - stime
        client_A.task_1("B","C")
        client_B.task_2("C")
        # stime = time.time()
        client_C.task_2("A", "B")
        client_A.task_2()
        # etime = time.time()
        # t += etime - stime
        client_B.task_3()

        # 预测
        y_pred = XA_test.dot(client_A.weights) + XB_test.dot(client_B.weights)
        normalized_arr = scaled_sigmoid(y_pred)

        # 分类决策
        y_pred_class = np.where(normalized_arr > 0, 1, -1)

        # 评估（假设 y_test 是 {-1,1}）
        acc = accuracy_score(y_test, y_pred_class)
        f1 = f1_score(y_test, y_pred_class, average='binary')
        print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    
    print("All process done.")


config = {
    'n_iter': 10,   #训练轮次
    'lambda': 10,   
    'lr': 0.05,     #学习率
    'A_idx': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  #a的特征空间
    'B_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],            #b的特征空间
}

X, y, X_test, y_test = load_data()                      #乳腺癌数据集，二分类任务，每个数据有30各数值型特征，目标变量是良性或恶性
vertical_logistic_regression(X, y, X_test, y_test, config)

