import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('regress_data1.csv')

# 提取人口和收益数据
population = data['人口']
profit = data['收益']



# 构造模型的损失（误差）计算函数
def cost(X, y, w):
    m = len(y)
    J = np.sum((X.dot(w) - y) ** 2) / (2 * m)
    return J

# 在人口数据中添加一列常数项，用于计算截距
population = population[:, np.newaxis]
profit = profit[:, np.newaxis]
ones = np.ones_like(population)
X = np.hstack((ones, population))

# 初始化模型参数
w = np.zeros((2, 1))


#梯度下降法
learning_rate = 0.01
num_iterations = 1000

def gradient_descent(X, y, w, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        # 计算预测值
        y_pred = np.dot(X, w)
        # 计算误差
        error = y_pred - y
        # 计算梯度
        gradient = np.dot(X.T, error) / m
        # 更新参数
        w -= learning_rate * gradient
        # 计算损失
        cost = np.sum((error ** 2)) / (2 * m)
        cost_history[i] = cost

    return w, cost_history

# 运行梯度下降算法
w, cost_history = gradient_descent(X, profit, w, learning_rate, num_iterations)

print("最终损失值：",cost_history[-1])

# 输出最终参数值
print("最优参数值：",end='\n')
print("w:",w[1],end='\n')
print("b:",w[0],end='\n')



# 画出模型拟合的直线和原始数据散点图
plt.figure()
plt.scatter(population, profit, marker='x', color='r')
plt.plot(population, np.dot(X, w), color='b')
plt.title('Linear regression fit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()