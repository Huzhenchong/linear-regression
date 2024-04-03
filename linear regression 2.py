import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('regress_data2.csv')
# 提取特征和标签
X = data[['面积', '房间数']].values
y = data['价格'].values

# 特征归一化
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

# 添加常数项列
X_norm = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))

# 初始化模型参数
w = np.zeros((3, 1))

# 批量梯度下降法
learning_rate = 0.01
num_iterations = 1000

def gradient_descent(X, y, w, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        y_pred = np.dot(X, w)
        error = y_pred - y.reshape(-1, 1)
        gradient = np.dot(X.T, error) / m
        w -= learning_rate * gradient
        cost = np.sum((error ** 2)) / (2 * m)
        cost_history[i] = cost

    return w, cost_history

# 运行梯度下降算法
w, cost_history = gradient_descent(X_norm, y, w, learning_rate, num_iterations)

# 输出最终参数值和损失值
print("最优参数值：",end='\n')
print("w1(房间数的参数):",w[-1],end='\n')
print("w2(面积的参数):",w[-2],end='\n')
print("b:",w[0],end='\n')
print("损失值：", cost_history[-1])

# 画出训练误差随着迭代轮数的变化曲线
plt.figure()
plt.plot(range(num_iterations), cost_history, color='b')
plt.title('Training Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.show()