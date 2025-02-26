import numpy as np


# ----------------------------- 卷积神经网络（CNN）算法 -----------------------------

# 介绍：
# 卷积神经网络（CNN）是一种用于处理具有网格状拓扑的数据的深度神经网络（如图像）。CNN通过
# 层叠卷积层、池化层和全连接层来自动提取特征。卷积层能够有效捕捉局部特征，而池化层则
# 用于下采样和特征缩放。CNN通常用于图像分类、物体检测等任务。

# 输入输出：
# 输入：
# - X: 输入数据，通常是图像数据，形状为 (n_samples, height, width, channels)。
# - y: 目标标签，形状为 (n_samples, n_classes)。
# 输出：
# - 模型训练好的权重和偏置，以及预测结果。

# 算法步骤：
# 1. 初始化卷积层、池化层和全连接层的权重和偏置。
# 2. 前向传播：依次计算卷积层、池化层和全连接层的输出。
# 3. 计算损失函数（交叉熵损失或均方误差）。
# 4. 反向传播：计算损失对每一层的梯度，更新权重和偏置。
# 5. 重复步骤2到步骤4，直到损失收敛。

# 主要参数：
# - learning_rate：学习率，用于权重更新的步长。
# - max_iter：最大迭代次数。
# - filter_size：卷积核的大小。
# - num_filters：每一层的卷积核个数。
# - pool_size：池化层的大小。

class CNN:
    def __init__(self, filter_size=3, num_filters=[32, 64], pool_size=2, learning_rate=0.001, max_iter=1000):
        """
        初始化卷积神经网络（CNN）模型。

        :param filter_size: 卷积核的大小。
        :param num_filters: 每一层卷积层的滤波器数量。
        :param pool_size: 池化层的大小。
        :param learning_rate: 学习率。
        :param max_iter: 最大迭代次数。
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.biases = []

    def _initialize_weights(self, input_shape):
        """
        初始化卷积层、池化层和全连接层的权重和偏置。

        :param input_shape: 输入数据的形状 (height, width, channels)。
        """
        height, width, channels = input_shape
        for num_filter in self.num_filters:
            weight = np.random.randn(self.filter_size, self.filter_size, channels, num_filter) * 0.01
            bias = np.zeros((1, num_filter))
            self.weights.append(weight)
            self.biases.append(bias)
            channels = num_filter  # 传递给下一层的通道数

        # 初始化全连接层权重
        self.fc_weights = np.random.randn(
            channels * (height // (2 ** len(self.num_filters))) * (width // (2 ** len(self.num_filters))),
            10) * 0.01  # 假设10个输出类
        self.fc_biases = np.zeros((1, 10))

    def _conv2d(self, X, W, b):
        """
        卷积运算（单个卷积层）。

        :param X: 输入数据，形状为 (batch_size, height, width, channels)。
        :param W: 卷积核，形状为 (filter_height, filter_width, in_channels, out_channels)。
        :param b: 偏置项，形状为 (1, out_channels)。
        :return: 卷积结果。
        """
        batch_size, height, width, channels = X.shape
        filter_height, filter_width, _, out_channels = W.shape
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1

        Z = np.zeros((batch_size, output_height, output_width, out_channels))

        for i in range(output_height):
            for j in range(output_width):
                region = X[:, i:i + filter_height, j:j + filter_width, :]
                Z[:, i, j, :] = np.tensordot(region, W, axes=((1, 2, 3), (0, 1, 2))) + b

        return Z

    def _relu(self, Z):
        """
        ReLU 激活函数。

        :param Z: 输入数据。
        :return: 激活后的输出。
        """
        return np.maximum(0, Z)

    def _pool(self, X, pool_size):
        """
        池化操作（最大池化）。

        :param X: 输入数据，形状为 (batch_size, height, width, channels)。
        :param pool_size: 池化窗口大小。
        :return: 池化后的输出。
        """
        batch_size, height, width, channels = X.shape
        output_height = height // pool_size
        output_width = width // pool_size

        Z = np.zeros((batch_size, output_height, output_width, channels))

        for i in range(output_height):
            for j in range(output_width):
                region = X[:, i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size, :]
                Z[:, i, j, :] = np.max(region, axis=(1, 2))

        return Z

    def _forward(self, X):
        """
        前向传播：计算卷积层、池化层和全连接层的输出。

        :param X: 输入数据。
        :return: 各层的输出。
        """
        activations = [X]

        # 卷积层和ReLU激活
        for i in range(len(self.weights)):
            Z = self._conv2d(activations[-1], self.weights[i], self.biases[i])
            activation = self._relu(Z)
            activations.append(activation)
            # 池化层
            pool = self._pool(activation, self.pool_size)
            activations.append(pool)

        # 将卷积和池化后的数据展平为一维
        flattened = activations[-1].reshape(activations[-1].shape[0], -1)

        # 全连接层
        fc_output = np.dot(flattened, self.fc_weights) + self.fc_biases
        activations.append(fc_output)

        return activations

    def _softmax(self, Z):
        """
        Softmax 激活函数，用于输出层。

        :param Z: 输入数据。
        :return: softmax结果。
        """
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _compute_loss(self, y_pred, y):
        """
        计算交叉熵损失。

        :param y_pred: 模型预测结果。
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

        # 反向传播输出层
        output = self._softmax(activations[-1])
        dz = output - y
        dw_fc = np.dot(activations[-2].T, dz) / m
        db_fc = np.sum(dz, axis=0, keepdims=True) / m
        self.fc_weights -= self.learning_rate * dw_fc
        self.fc_biases -= self.learning_rate * db_fc

        # 反向传播池化层和卷积层
        dz = np.dot(dz, self.fc_weights.T)
        dz = dz.reshape(activations[-2].shape)

        for i in range(len(self.weights) - 1, -1, -1):
            pool = activations[2 * i + 1]
            dz = dz * (pool > 0)  # ReLU梯度
            dw_conv = np.zeros_like(self.weights[i])
            db_conv = np.zeros_like(self.biases[i])
            for j in range(dz.shape[0]):
                dw_conv += np.tensordot(pool[j], dz[j], axes=((0, 1, 2), (0, 1, 2)))
                db_conv += np.sum(dz[j], axis=(0, 1))
            self.weights[i] -= self.learning_rate * dw_conv / m
            self.biases[i] -= self.learning_rate * db_conv / m

    def fit(self, X, y):
        """
        训练卷积神经网络模型。

        :param X: 输入数据。
        :param y: 目标标签。
        """
        self._initialize_weights(X.shape[1:])

        for _ in range(self.max_iter):
            activations = self._forward(X)
            loss = self._compute_loss(self._softmax(activations[-1]), y)
            self._backward(activations, X, y)
            print(f"Loss: {loss}")

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        :param X: 输入数据。
        :return: 预测结果。
        """
        activations = self._forward(X)
        return self._softmax(activations[-1])


# 示例：创建CNN模型并训练
if __name__ == "__main__":
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8, 1)  # 转换为图像格式

    # One-hot编码标签
    y = np.eye(10)[y]

    cnn = CNN(filter_size=3, num_filters=[32, 64], pool_size=2, learning_rate=0.001, max_iter=100)
    cnn.fit(X, y)