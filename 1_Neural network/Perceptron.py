import numpy as np


# ----------------------------- 感知机（Perceptron）算法 -----------------------------

# 介绍：
# 感知机（Perceptron）是最简单的线性分类模型之一，通常用于二分类问题。感知机通过
# 加权和的方式，将输入特征映射到一个线性空间，并通过激活函数（通常为阶跃函数）来
# 预测类别。感知机模型的目标是通过不断调整权重和偏置，使得所有训练样本能够被正确分类。

# 输入输出：
# 输入：
# - X: 训练数据的特征矩阵，形状为 (n_samples, n_features)。
# - y: 训练数据的标签，形状为 (n_samples,)，值为 -1 或 1。
# 输出：
# - 训练完成的感知机模型，其中包括更新后的权重和偏置。

# 算法步骤：
# 1. 初始化权重（w）和偏置（b）。权重初始化为零。
# 2. 对每个训练样本计算感知机输出：linear_output = w * X + b。
# 3. 根据预测值与真实标签的比较来更新权重和偏置：
#    - 若预测正确，则不更新；
#    - 若预测错误，使用公式 w = w + learning_rate * y * X 更新权重，b = b + learning_rate * y 更新偏置。
# 4. 重复以上过程，直到所有样本被正确分类或达到最大迭代次数。

# 主要参数：
# - learning_rate：控制每次权重更新的步长。
# - max_iter：最大迭代次数，用于控制训练过程的最大轮数。

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        """
        初始化感知机模型。

        :param learning_rate: 学习率，控制权重更新的步长。
        :param max_iter: 最大迭代次数，限制训练的次数。
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练感知机模型，通过多次迭代来更新权重和偏置。

        :param X: 训练数据特征矩阵，形状为 (n_samples, n_features)。
        :param y: 训练数据标签，形状为 (n_samples,)，值为 -1 或 1。
        """
        # 初始化权重和偏置
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 迭代训练
        for _ in range(self.max_iter):
            # 标志位：是否本次迭代更新了权重
            updated = False
            for i in range(n_samples):
                # 计算感知机的预测值
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = np.sign(linear_output)

                # 更新权重和偏置（若预测错误）
                if prediction != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    updated = True

            # 如果本轮没有更新权重，说明模型已经收敛，提前终止
            if not updated:
                break

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        :param X: 输入的特征数据，形状为 (n_samples, n_features)。
        :return: 预测结果，形状为 (n_samples,)。
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

    def score(self, X, y):
        """
        计算模型的准确率。

        :param X: 测试数据特征。
        :param y: 测试数据标签。
        :return: 模型的准确率。
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def plot_decision_boundary(self, X, y):
        """
        可视化决策边界，仅适用于2D特征。

        :param X: 输入数据，形状为 (n_samples, 2)。
        :param y: 数据标签。
        """
        import matplotlib.pyplot as plt
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
        plt.title("Perceptron Decision Boundary")
        plt.show()


# 示例：使用感知机算法训练并测试
if __name__ == "__main__":
    # 生成一个简单的线性可分数据集
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    y = 2 * y - 1  # 将标签转换为 -1 和 1

    # 训练感知机模型
    perceptron = Perceptron(learning_rate=0.1, max_iter=1000)
    perceptron.fit(X, y)

    # 测试模型
    accuracy = perceptron.score(X, y)
    print(f"训练集上的准确率: {accuracy * 100:.2f}%")

    # 可视化决策边界
    perceptron.plot_decision_boundary(X, y)