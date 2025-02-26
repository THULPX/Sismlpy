import numpy as np


# ----------------------------- 变分自编码器（VAE）算法 -----------------------------

# 介绍：
# 变分自编码器（Variational Autoencoder, VAE）是一种生成模型，利用变分推断方法对自编码器进行扩展。VAE通过优化变分下界
# (ELBO) 来学习数据的潜在空间分布。在VAE中，编码器生成潜在变量的分布，而解码器从该分布中采样生成数据。VAE可用于图像生成、数据重构、去噪等任务。

# 输入输出：
# 输入：
# - X: 输入数据，形状为 (n_samples, input_dim)。
# 输出：
# - 重构的输出数据。
# - 潜在空间的样本。

# 算法步骤：
# 1. 编码器将输入数据映射到潜在空间的分布（均值和方差）。
# 2. 从潜在空间的分布中采样得到潜在变量。
# 3. 解码器从潜在变量生成重构数据。
# 4. 通过最大化变分下界（ELBO）来优化模型参数，包含重构误差和KL散度。

# 主要参数：
# - input_dim: 输入数据的维度。
# - latent_dim: 潜在空间的维度。
# - learning_rate: 学习率。
# - max_iter: 最大训练迭代次数。

class VAE:
    def __init__(self, input_dim, latent_dim, learning_rate=0.001, max_iter=1000):
        """
        初始化变分自编码器（VAE）模型。

        :param input_dim: 输入数据的维度。
        :param latent_dim: 潜在空间的维度。
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

    def _sampling(self, mean, log_var):
        """
        从均值和对数方差中进行重参数化采样。

        :param mean: 潜在变量的均值。
        :param log_var: 潜在变量的对数方差。
        :return: 潜在变量的采样值。
        """
        epsilon = np.random.randn(*mean.shape)
        return mean + np.exp(0.5 * log_var) * epsilon

    def _encode(self, X):
        """
        编码器：将输入数据映射到潜在空间的分布（均值和对数方差）。

        :param X: 输入数据。
        :return: 潜在空间的均值和对数方差。
        """
        Z_encoded = np.dot(X, self.W_encoder) + self.b_encoder
        mean = Z_encoded[:, :self.latent_dim]
        log_var = Z_encoded[:, self.latent_dim:]
        return mean, log_var

    def _decode(self, z):
        """
        解码器：将潜在空间的样本解码为重构数据。

        :param z: 潜在空间的样本。
        :return: 重构的数据。
        """
        decoded = self._sigmoid(np.dot(z, self.W_decoder) + self.b_decoder)
        return decoded

    def _compute_loss(self, X, decoded, mean, log_var):
        """
        计算VAE的损失函数，包括重构误差和KL散度。

        :param X: 输入数据。
        :param decoded: 重构的输出数据。
        :param mean: 潜在空间的均值。
        :param log_var: 潜在空间的对数方差。
        :return: VAE的损失值。
        """
        # 重构误差（均方误差）
        reconstruction_loss = np.mean((X - decoded) ** 2)

        # KL散度
        kl_divergence = -0.5 * np.mean(1 + log_var - np.square(mean) - np.exp(log_var))

        # 总损失
        loss = reconstruction_loss + kl_divergence
        return loss, reconstruction_loss, kl_divergence

    def _backward(self, X, decoded, mean, log_var, z):
        """
        反向传播：计算梯度并更新权重和偏置。

        :param X: 输入数据。
        :param decoded: 重构的输出数据。
        :param mean: 潜在空间的均值。
        :param log_var: 潜在空间的对数方差。
        :param z: 潜在空间的样本。
        """
        m = X.shape[0]

        # 重构误差对解码器的梯度
        dZ_decoded = decoded - X
        dW_decoder = np.dot(z.T, dZ_decoded) / m
        db_decoder = np.sum(dZ_decoded, axis=0, keepdims=True) / m

        # KL散度对编码器的梯度
        dmean = -mean / np.exp(log_var)  # KL散度对均值的梯度
        dlog_var = 0.5 * (np.exp(log_var) - 1 / np.exp(log_var))  # KL散度对对数方差的梯度
        dW_encoder = np.dot(X.T, np.concatenate([dmean, dlog_var], axis=1)) / m
        db_encoder = np.sum(np.concatenate([dmean, dlog_var], axis=1), axis=0, keepdims=True) / m

        # 更新权重和偏置
        self.W_encoder -= self.learning_rate * dW_encoder
        self.b_encoder -= self.learning_rate * db_encoder
        self.W_decoder -= self.learning_rate * dW_decoder
        self.b_decoder -= self.learning_rate * db_decoder

    def fit(self, X):
        """
        训练变分自编码器（VAE）模型。

        :param X: 输入数据。
        """
        for epoch in range(self.max_iter):
            # 编码器：得到均值和对数方差
            mean, log_var = self._encode(X)

            # 从均值和对数方差中采样得到潜在变量
            z = self._sampling(mean, log_var)

            # 解码器：从潜在空间采样生成重构数据
            decoded = self._decode(z)

            # 计算损失
            loss, reconstruction_loss, kl_divergence = self._compute_loss(X, decoded, mean, log_var)

            # 反向传播：更新权重和偏置
            self._backward(X, decoded, mean, log_var, z)

            # 输出每个epoch的损失
            print(
                f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss}, Reconstruction Loss: {reconstruction_loss}, KL Divergence: {kl_divergence}")

    def predict(self, X):
        """
        使用训练好的VAE进行数据重构。

        :param X: 输入数据。
        :return: 重构的输出数据。
        """
        mean, log_var = self._encode(X)
        z = self._sampling(mean, log_var)
        return self._decode(z)


# 示例：创建VAE模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据（例如，二维数据）
    X = np.random.randn(100, 3)  # 100个样本，3维输入

    # 创建VAE模型
    vae = VAE(input_dim=3, latent_dim=2, learning_rate=0.001, max_iter=1000)

    # 训练VAE模型
    vae.fit(X)

    # 使用训练好的VAE进行数据重构
    X_reconstructed = vae.predict(X)
    print("原始数据：", X[:5])
    print("重构数据：", X_reconstructed[:5])
