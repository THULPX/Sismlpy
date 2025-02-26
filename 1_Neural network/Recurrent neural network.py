import numpy as np


# ----------------------------- 循环神经网络（RNN）算法 -----------------------------

# 介绍：
# 循环神经网络（RNN）是一类用于处理序列数据的神经网络，广泛应用于自然语言处理、时间序列分析等领域。
# RNN通过共享参数的方式来处理任意长度的序列数据，可以根据前一个时间步的输出和当前输入进行计算。
# 但传统的RNN在长序列中存在梯度消失或爆炸问题，因此常常使用LSTM或GRU等变体。

# 输入输出：
# 输入：
# - X: 输入序列数据，形状为 (n_samples, sequence_length, input_size)。
# - y: 目标标签，形状为 (n_samples, output_size)。
# 输出：
# - 模型训练好的权重和偏置，以及预测结果。

# 算法步骤：
# 1. 初始化RNN模型的权重和偏置。
# 2. 前向传播：依次计算每个时间步的隐藏状态。
# 3. 计算损失函数（均方误差或交叉熵损失）。
# 4. 反向传播：计算损失对每个参数的梯度并更新权重。
# 5. 重复步骤2到步骤4，直到损失收敛。

# 主要参数：
# - learning_rate：学习率，用于权重更新的步长。
# - max_iter：最大迭代次数。
# - hidden_size：隐藏层的大小。
# - output_size：输出的维度。
# - input_size：输入的维度。

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, max_iter=1000):
        """
        初始化循环神经网络（RNN）模型。

        :param input_size: 输入的维度。
        :param hidden_size: 隐藏层的大小。
        :param output_size: 输出的维度。
        :param learning_rate: 学习率。
        :param max_iter: 最大迭代次数。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # 初始化权重和偏置
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01  # 输入到隐藏的权重
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏到隐藏的权重
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏到输出的权重
        self.bh = np.zeros((1, hidden_size))  # 隐藏层的偏置
        self.by = np.zeros((1, output_size))  # 输出层的偏置

    def _sigmoid(self, Z):
        """
        Sigmoid 激活函数。

        :param Z: 输入数据。
        :return: Sigmoid 激活后的输出。
        """
        return 1 / (1 + np.exp(-Z))

    def _tanh(self, Z):
        """
        Tanh 激活函数。

        :param Z: 输入数据。
        :return: Tanh 激活后的输出。
        """
        return np.tanh(Z)

    def _softmax(self, Z):
        """
        Softmax 激活函数。

        :param Z: 输入数据。
        :return: Softmax 激活后的输出。
        """
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _forward(self, X):
        """
        前向传播：计算每个时间步的隐藏状态。

        :param X: 输入数据。
        :return: 每个时间步的隐藏状态和最终输出。
        """
        batch_size, sequence_length, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))  # 隐藏状态初始化为零
        outputs = []

        for t in range(sequence_length):
            # 输入到隐藏层的计算
            h = self._tanh(np.dot(X[:, t, :], self.Wx) + np.dot(h, self.Wh) + self.bh)
            outputs.append(h)

        # 计算最终输出
        y_pred = np.dot(h, self.Wy) + self.by
        return np.array(outputs), y_pred

    def _compute_loss(self, y_pred, y):
        """
        计算损失函数（均方误差）。

        :param y_pred: 模型的预测结果。
        :param y: 真实标签。
        :return: 损失值。
        """
        m = y.shape[0]
        return np.mean((y_pred - y) ** 2)

    def _backward(self, X, y, outputs, y_pred):
        """
        反向传播：计算每个时间步的梯度并更新权重。

        :param X: 输入数据。
        :param y: 真实标签。
        :param outputs: 每个时间步的隐藏状态。
        :param y_pred: 模型的预测结果。
        """
        m = y.shape[0]
        dWy = np.dot(outputs[-1].T, (y_pred - y)) / m  # 输出层的梯度
        dby = np.sum(y_pred - y, axis=0, keepdims=True) / m  # 输出层的偏置梯度

        # 反向传播隐藏层
        dh = np.dot(y_pred - y, self.Wy.T)

        for t in reversed(range(len(outputs))):
            dh_raw = dh * (1 - outputs[t] ** 2)  # Tanh 激活函数的梯度
            dWx = np.dot(X[:, t, :].T, dh_raw) / m
            dWh = np.dot(outputs[t - 1].T, dh_raw) / m if t > 0 else np.zeros_like(dh_raw)
            dbh = np.sum(dh_raw, axis=0, keepdims=True) / m

            # 更新权重
            self.Wx -= self.learning_rate * dWx
            self.Wh -= self.learning_rate * dWh
            self.bh -= self.learning_rate * dbh

            # 计算梯度传递给上一个时间步
            dh = np.dot(dh_raw, self.Wh.T)

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def fit(self, X, y):
        """
        训练循环神经网络模型。

        :param X: 输入序列数据。
        :param y: 目标标签。
        """
        for _ in range(self.max_iter):
            outputs, y_pred = self._forward(X)
            loss = self._compute_loss(y_pred, y)
            self._backward(X, y, outputs, y_pred)
            print(f"Loss: {loss}")

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        :param X: 输入数据。
        :return: 预测结果。
        """
        _, y_pred = self._forward(X)
        return y_pred


# 示例：创建RNN模型并训练
if __name__ == "__main__":
    # 生成一个简单的时间序列数据
    X = np.random.randn(100, 10, 5)  # 100个样本，序列长度10，每个时间步5维输入
    y = np.random.randn(100, 1)  # 100个标签，1维输出

    rnn = RNN(input_size=5, hidden_size=50, output_size=1, learning_rate=0.001, max_iter=100)
    rnn.fit(X, y)
