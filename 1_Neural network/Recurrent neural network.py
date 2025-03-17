import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, max_iter=1000):
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
        self.losses = []  # 存储训练过程中的损失值

        # 使用Xavier初始化来改善梯度流动
        self.Wx = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.Wh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def _relu(self, Z):
        """
        ReLU激活函数。

        :param Z: 输入数据。
        :return: ReLU激活后的输出。
        """
        return np.maximum(0, Z)

    def _compute_loss(self, y_pred, y):
        """
        计算损失函数（均方误差）。

        :param y_pred: 模型的预测结果。
        :param y: 真实标签。
        :return: 损失值。
        """
        return np.mean((y_pred - y) ** 2)  # 使用MSE而不是交叉熵，因为这是回归问题

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
            # ReLU激活函数，避免梯度消失
            h_temp = np.dot(X[:, t, :], self.Wx) + np.dot(h, self.Wh) + self.bh
            h = self._relu(h_temp)  # ReLU activation
            outputs.append(h)

        # 最后一层使用线性激活，适合回归问题
        y_pred = np.dot(h, self.Wy) + self.by
        return np.array(outputs), y_pred

    def _backward(self, X, y, outputs, y_pred):
        """
        反向传播：计算每个时间步的梯度并更新权重。

        :param X: 输入数据。
        :param y: 真实标签。
        :param outputs: 每个时间步的隐藏状态。
        :param y_pred: 模型的预测结果。
        """
        batch_size = y.shape[0]
        sequence_length = len(outputs)
        
        # 输出层的梯度计算
        dy = 2 * (y_pred - y) / batch_size  # MSE的导数
        dWy = np.dot(outputs[-1].T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)

        # 初始化隐藏层的梯度
        dh = np.dot(dy, self.Wy.T)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        # 反向传播通过时间步
        for t in reversed(range(sequence_length)):
            # ReLU的导数：大于0的部分为1，小于0的部分为0
            dh_raw = dh * (outputs[t] > 0).astype(float)
            
            # 累积各个参数的梯度
            dWx += np.dot(X[:, t, :].T, dh_raw)
            if t > 0:
                dWh += np.dot(outputs[t-1].T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)

            # 为下一个时间步计算隐藏状态的梯度
            if t > 0:
                dh = np.dot(dh_raw, self.Wh.T)

        # 更新所有参数，使用动量来加速训练
        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.Wy -= self.learning_rate * dWy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def fit(self, X, y):
        """
        训练循环神经网络模型。

        :param X: 输入序列数据。
        :param y: 目标标签。
        :return: 训练历史
        """
        # 数据标准化
        X = (X - np.mean(X)) / np.std(X)
        y = (y - np.mean(y)) / np.std(y)
        
        self.losses = []  # 重置损失记录
        for epoch in range(self.max_iter):
            outputs, y_pred = self._forward(X)
            loss = self._compute_loss(y_pred, y)
            self.losses.append(loss)
            self._backward(X, y, outputs, y_pred)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
            # 提前停止条件
            #if len(self.losses) > 2 and abs(self.losses[-1] - self.losses[-2]) < 1e-6:
                #print("收敛，提前停止训练")
                #break
        
        return self.losses

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
    np.random.seed(42)  # 设置随机种子以便复现
    X = np.random.randn(100, 10, 5)  # 100个样本，序列长度10，每个时间步5维输入
    # 生成一个更有意义的目标函数：y是X序列的加权和
    weights = np.random.randn(5)
    y = np.sum(np.mean(X, axis=1) * weights, axis=1, keepdims=True)
    
    # 创建并训练模型
    rnn = RNN(input_size=5, hidden_size=32, output_size=1, learning_rate=0.01, max_iter=2000)
    losses = rnn.fit(X, y)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')  # 使用对数坐标以更好地显示损失变化
    plt.grid(True)
    
    # 绘制预测值与真实值的对比
    y_pred = rnn.predict(X)
    plt.subplot(1, 2, 2)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 理想预测线
    plt.title('Predicted value VS True value')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算并打印评估指标
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"\n模型评估:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"R² 分数: {r2:.6f}")
