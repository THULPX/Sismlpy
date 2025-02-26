import numpy as np


# ----------------------------- 自编码器（Autoencoder）算法 -----------------------------

# 介绍：
# 自编码器（Autoencoder）是一种神经网络模型，旨在学习输入数据的低维表示。自编码器由编码器（Encoder）和解码器（Decoder）两部分组成，
# 编码器将输入数据映射到低维空间（隐空间），解码器则试图从这个低维空间重构出原始数据。自编码器在降维、去噪和生成模型等领域有广泛应用。

# 输入输出：
# 输入：
# - X: 输入数据，形状为 (n_samples, input_dim)。
# 输出：
# - 重构的输出数据。

# 算法步骤：
# 1. 构建自编码器模型，包括编码器和解码器。
# 2. 编码器将输入数据压缩成一个低维隐空间表示。
# 3. 解码器从隐空间表示中重构出原始输入数据。
# 4. 使用损失函数（例如均方误差）来训练模型，最小化输入数据和重构数据之间的差异。

# 主要参数：
# - input_dim: 输入数据的维度。
# - latent_dim: 隐空间的维度。
# - learning_rate: 学习率。
# - max_iter: 最大训练迭代次数。

class Autoencoder:
    def __init__(self, input_dim, latent_dim, learning_rate=0.001, max_iter=1000):
        """
        初始化自编码器模型。

        :param input_dim: 输入数据的维度。
        :param latent_dim: 隐空间的维度。
        :param learning_rate: 学习率。
        :param max_iter: 最大训练迭代次数。
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # 初始化编码器和解码器的权重和偏置
        self.W_encoder = np.random.randn(input_dim, latent_dim) * 0.01
        self.b_encoder = np.zeros((1, latent_dim))
        self.W_decoder = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_decoder = np.zeros((1, input_dim))

    def _relu(self, Z):
        """
        ReLU 激活函数。

        :param Z: 输入数据。
        :return: ReLU 激活后的输出。
        """
        return np.maximum(0, Z)

    def _sigmoid(self, Z):
        """
        Sigmoid 激活函数。

        :param Z: 输入数据。
        :return: Sigmoid 激活后的输出。
        """
        return 1 / (1 + np.exp(-Z))

    def _forward(self, X):
        """
        前向传播：通过编码器和解码器生成重构的输出。

        :param X: 输入数据。
        :return: 重构的输出。
        """
        # 编码器：将输入数据映射到隐空间
        Z_encoded = np.dot(X, self.W_encoder) + self.b_encoder
        encoded = self._relu(Z_encoded)

        # 解码器：将隐空间表示解码回原始空间
        Z_decoded = np.dot(encoded, self.W_decoder) + self.b_decoder
        decoded = self._sigmoid(Z_decoded)  # 通常使用sigmoid激活来处理重构的输出
        return decoded

    def _compute_loss(self, X, decoded):
        """
        计算重构误差（均方误差）。

        :param X: 输入数据。
        :param decoded: 重构的输出。
        :return: 均方误差损失。
        """
        return np.mean((X - decoded) ** 2)

    def _backward(self, X, decoded):
        """
        反向传播：计算梯度并更新权重和偏置。

        :param X: 输入数据。
        :param decoded: 重构的输出。
        """
        m = X.shape[0]

        # 计算重构误差对解码器的梯度
        dZ_decoded = decoded - X
        dW_decoder = np.dot(decoded.T, dZ_decoded) / m
        db_decoder = np.sum(dZ_decoded, axis=0, keepdims=True) / m

        # 计算解码器输出对编码器的梯度
        dencoded = np.dot(dZ_decoded, self.W_decoder.T)
        dZ_encoded = dencoded * (encoded > 0)  # ReLU的导数

        # 计算编码器的梯度
        dW_encoder = np.dot(X.T, dZ_encoded) / m
        db_encoder = np.sum(dZ_encoded, axis=0, keepdims=True) / m

        # 更新权重和偏置
        self.W_encoder -= self.learning_rate * dW_encoder
        self.b_encoder -= self.learning_rate * db_encoder
        self.W_decoder -= self.learning_rate * dW_decoder
        self.b_decoder -= self.learning_rate * db_decoder

    def fit(self, X):
        """
        训练自编码器模型。

        :param X: 输入数据。
        """
        for epoch in range(self.max_iter):
            # 前向传播：获取重构输出
            decoded = self._forward(X)

            # 计算损失
            loss = self._compute_loss(X, decoded)

            # 反向传播：更新权重和偏置
            self._backward(X, decoded)

            # 输出每个epoch的损失
            print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss}")

    def predict(self, X):
        """
        使用训练好的自编码器进行数据重构。

        :param X: 输入数据。
        :return: 重构的输出。
        """
        return self._forward(X)


# 示例：创建自编码器模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据（例如，二维数据）
    X = np.random.randn(100, 3)  # 100个样本，3维输入

    # 创建自编码器模型
    autoencoder = Autoencoder(input_dim=3, latent_dim=2, learning_rate=0.001, max_iter=1000)

    # 训练自编码器
    autoencoder.fit(X)

    # 使用训练好的自编码器进行数据重构
    X_reconstructed = autoencoder.predict(X)
    print("原始数据：", X[:5])
    print("重构数据：", X_reconstructed[:5])
