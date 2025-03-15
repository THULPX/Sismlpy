import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

# -------------------------- 多层感知机（MLP）算法 --------------------------

# 介绍：
# 多层感知机（MLP）是一个前馈神经网络，包括输入层、一个或多个隐藏层和输出层。
# 每一层的神经元与前一层的神经元全连接，使用非线性激活函数进行映射。
# MLP通过反向传播算法进行训练，利用梯度下降法优化权重。

# 输入输出：
# 输入：
# - X: 输入数据的特征矩阵，形状为 (n_samples, n_features)。
# - y: 目标标签，形状为 (n_samples, n_outputs)。
# 输出：
# - 模型训练好的权重和偏置，以及预测结果。

# 算法步骤：
# 1. 随机初始化网络各层的权重和偏置。
# 2. 前向传播：计算每层加权输入，并通过激活函数得到输出。
# 3. 计算损失函数：根据网络的预测输出与实际标签之间的误差计算损失。
# 4. 反向传播：通过链式法则计算损失对每层参数的梯度，更新权重和偏置。
# 5. 重复前向传播、计算损失和反向传播，直到最大迭代次数或损失收敛。

# 主要参数：
# - learning_rate：学习率，控制更新步长。
# - max_iter：最大迭代次数。
# - hidden_layers：一个或多个隐藏层的神经元数量。

class MLP:
    def __init__(self, hidden_layers=[5], learning_rate=0.01, max_iter=5000, activation='relu',
                 output_activation='softmax'):
        """
        初始化多层感知机模型。

        :param hidden_layers: 隐藏层的神经元数量（可以是一个整数或列表）。
        :param learning_rate: 学习率。
        :param max_iter: 最大迭代次数。
        :param activation: 隐藏层激活函数（'relu', 'sigmoid', 'tanh'等）。
        :param output_activation: 输出层激活函数（'softmax', 'sigmoid', 'linear'等）。
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.losses = []

    def _initialize_weights(self, input_dim, output_dim):
        """
        初始化权重和偏置。

        :param input_dim: 输入层神经元数量。
        :param output_dim: 输出层神经元数量。
        """
        layer_dims = [input_dim] + self.hidden_layers + [output_dim]
        for i in range(1, len(layer_dims)):
            weight = np.random.randn(layer_dims[i - 1], layer_dims[i]) * 0.01
            bias = np.zeros((1, layer_dims[i]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _activation(self, z, activation_type):
        """
        激活函数。

        :param z: 输入的加权和。
        :param activation_type: 激活函数类型（'relu', 'sigmoid', 'tanh'等）。
        :return: 激活后的输出。
        """
        if activation_type == 'relu':
            return np.maximum(0, z)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation_type == 'tanh':
            return np.tanh(z)
        return z

    def _output_activation(self, z, activation_type):
        """
        输出层激活函数。

        :param z: 输入的加权和。
        :param activation_type: 输出层激活函数类型（'softmax', 'sigmoid', 'linear'等）。
        :return: 激活后的输出。
        """
        if activation_type == 'softmax':
            exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        return z

    def _forward(self, X):
        """
        前向传播：计算每层的输出。

        :param X: 输入数据。
        :return: 各层的输出值。
        """
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self._activation(z, self.activation) if i < len(self.weights) - 1 else self._output_activation(
                z, self.output_activation)
            activations.append(activation)
        return activations

    def _compute_loss(self, y_pred, y):
        """
        计算损失函数（交叉熵损失）。

        :param y_pred: 网络的预测输出。
        :param y: 真实标签。
        :return: 损失值。
        """
        m = y.shape[0]
        return -np.sum(y * np.log(y_pred + 1e-8)) / m

    def _backward(self, activations, X, y):
        """
        反向传播：计算每层的梯度并更新权重和偏置。

        :param activations: 前向传播得到的各层激活值。
        :param X: 输入数据。
        :param y: 真实标签。
        """
        m = X.shape[0]
        dz = activations[-1] - y
        dw = np.dot(activations[-2].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        self.weights[-1] -= self.learning_rate * dw
        self.biases[-1] -= self.learning_rate * db

        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * (activations[i + 1] > 0)
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y):
        """
        训练多层感知机模型。

        :param X: 输入数据特征矩阵。
        :param y: 真实标签。
        """
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        self._initialize_weights(input_dim, output_dim)

        for _ in range(self.max_iter):
            activations = self._forward(X)
            loss = self._compute_loss(activations[-1], y)
            self.losses.append(loss)
            self._backward(activations, X, y)

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        :param X: 输入数据特征矩阵。
        :return: 模型的预测输出。
        """
        activations = self._forward(X)
        return activations[-1]

    def score(self, X, y):
        """
        计算模型的准确率。

        :param X: 测试数据特征。
        :param y: 测试数据标签。
        :return: 准确率。
        """
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)


# 示例：使用MLP进行训练
if __name__ == "__main__":
    # 生成一个稍微复杂的分类数据集，增加特征和类别
    X, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, random_state=42)
    y = np.eye(3)[y]  # 转换为one-hot编码

    # 创建MLP模型并训练，增加了更多的隐藏层和节点
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), learning_rate_init=0.01, max_iter=10000, activation='relu', solver='adam')
    mlp.fit(X, y)

    # 打印训练后的权重和偏置
    print("训练后的权重和偏置：")
    for i, (weight, bias) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        print(f"层 {i + 1} - 权重形状: {weight.shape}, 偏置形状: {bias.shape}")

    # 训练集准确率
    accuracy = mlp.score(X, y)
    print(f"训练集上的准确率: {accuracy * 100:.2f}%")

    # 绘制损失曲线
    plt.plot(mlp.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # 绘制决策边界
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape)])  # 三维数据，填充Z轴为0
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='k', marker='o', s=50, cmap=plt.cm.RdYlBu)
    plt.title("Decision Boundary")
    plt.show()