import numpy as np


# ----------------------------- 长短时记忆网络（LSTM）算法 -----------------------------

# 介绍：
# 长短时记忆网络（LSTM）是一种改进版的循环神经网络（RNN），通过引入“门”机制（遗忘门、输入门、输出门）
# 来解决传统RNN在长序列中梯度消失和梯度爆炸的问题。LSTM能够学习长期的依赖关系，广泛应用于
# 语言模型、时间序列预测、语音识别等任务。

# 输入输出：
# 输入：
# - X: 输入序列数据，形状为 (n_samples, sequence_length, input_size)。
# - y: 目标标签，形状为 (n_samples, output_size)。
# 输出：
# - 模型训练好的权重和偏置，以及预测结果。

# 算法步骤：
# 1. 初始化LSTM模型的权重和偏置。
# 2. 前向传播：计算每个时间步的细胞状态和隐藏状态。
# 3. 计算损失函数（均方误差或交叉熵损失）。
# 4. 反向传播：计算损失对每个参数的梯度并更新权重。
# 5. 重复步骤2到步骤4，直到损失收敛。

# 主要参数：
# - learning_rate：学习率，用于权重更新的步长。
# - max_iter：最大迭代次数。
# - hidden_size：隐藏层的大小。
# - output_size：输出的维度。
# - input_size：输入的维度。

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, max_iter=1000):
        """
        初始化长短时记忆网络（LSTM）模型。

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

        # 初始化LSTM模型的权重和偏置
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # 遗忘门权重
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # 输入门权重
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # 细胞状态权重
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # 输出门权重
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01  # 输出层权重

        self.bf = np.zeros((1, hidden_size))  # 遗忘门偏置
        self.bi = np.zeros((1, hidden_size))  # 输入门偏置
        self.bc = np.zeros((1, hidden_size))  # 细胞状态偏置
        self.bo = np.zeros((1, hidden_size))  # 输出门偏置
        self.by = np.zeros((1, output_size))  # 输出层偏置

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
        前向传播：计算每个时间步的LSTM状态（细胞状态和隐藏状态）。

        :param X: 输入数据。
        :return: 每个时间步的隐藏状态和最终输出。
        """
        batch_size, sequence_length, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))  # 初始化隐藏状态
        c = np.zeros((batch_size, self.hidden_size))  # 初始化细胞状态
        outputs = []

        for t in range(sequence_length):
            # 拼接输入和上一时刻的隐藏状态
            combined = np.hstack((X[:, t, :], h))

            # 遗忘门、输入门、细胞状态、输出门的计算
            ft = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
            it = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
            ct = np.tanh(np.dot(combined, self.Wc) + self.bc)
            ot = self._sigmoid(np.dot(combined, self.Wo) + self.bo)

            # 更新细胞状态
            c = ft * c + it * ct
            # 更新隐藏状态
            h = ot * np.tanh(c)

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

        # 反向传播隐藏层和LSTM门
        dh = np.dot(y_pred - y, self.Wy.T)
        dc = dh * np.tanh(outputs[-1]) * (1 - np.tanh(outputs[-1]) ** 2)  # 反向传播细胞状态

        for t in reversed(range(len(outputs))):
            combined = np.hstack((X[:, t, :], outputs[t - 1] if t > 0 else np.zeros_like(outputs[0])))
            ft = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
            it = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
            ct = np.tanh(np.dot(combined, self.Wc) + self.bc)
            ot = self._sigmoid(np.dot(combined, self.Wo) + self.bo)

            # 计算梯度
            dft = dc * c * ft * (1 - ft)
            dit = dc * ct * it * (1 - it)
            dct = dc * it * (1 - ct ** 2)
            dot = dc * np.tanh(c) * ot * (1 - ot)

            # 更新梯度
            dWf = np.dot(combined.T, dft) / m
            dWi = np.dot(combined.T, dit) / m
            dWc = np.dot(combined.T, dct) / m
            dWo = np.dot(combined.T, dot) / m
            dbf = np.sum(dft, axis=0, keepdims=True) / m
            dbi = np.sum(dit, axis=0, keepdims=True) / m
            dbc = np.sum(dct, axis=0, keepdims=True) / m
            dbo = np.sum(dot, axis=0, keepdims=True) / m

            # 更新权重
            self.Wf -= self.learning_rate * dWf
            self.Wi -= self.learning_rate * dWi
            self.Wc -= self.learning_rate * dWc
            self.Wo -= self.learning_rate * dWo
            self.by -= self.learning_rate * dbf
            self.bi -= self.learning_rate * dbi
            self.bc -= self.learning_rate * dbc
            self.bo -= self.learning_rate * dbo

            # 传递梯度到上一层
            dc = dc * ft

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def fit(self, X, y):
        """
        训练LSTM模型。

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


# 示例：创建LSTM模型并训练
if __name__ == "__main__":
    # 生成一个简单的时间序列数据
    X = np.random.randn(100, 10, 5)  # 100个样本，序列长度10，每个时间步5维输入
    y = np.random.randn(100, 1)  # 100个标签，1维输出

    lstm = LSTM(input_size=5, hidden_size=50, output_size=1, learning_rate=0.001, max_iter=100)
    lstm.fit(X, y)