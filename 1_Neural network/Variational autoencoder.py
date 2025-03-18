import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- 变分自编码器（VAE）算法 -----------------------------

# 介绍：
# 变分自编码器（Variational Autoencoder, VAE）是一种生成模型，利用变分推断方法对自编码器进行扩展。
# VAE通过优化变分下界(ELBO) 来学习数据的潜在空间分布。
# 在VAE中，编码器生成潜在变量的分布，而解码器从该分布中采样生成数据。
# VAE可用于图像生成、数据重构、去噪等任务。

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
    def __init__(self, input_dim, latent_dim, learning_rate=0.001, max_iter=5000):
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

        # 增加的隐藏层维度
        hidden_dim = (input_dim + latent_dim) // 2 * 2  # 增加隐藏层维度

        # 编码器权重
        self.W_encoder_h1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b_encoder_h1 = np.zeros((1, hidden_dim))
        self.W_encoder_h2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b_encoder_h2 = np.zeros((1, hidden_dim))
        self.W_encoder = np.random.randn(hidden_dim, 2 * latent_dim) / np.sqrt(hidden_dim)
        self.b_encoder = np.zeros((1, 2 * latent_dim))

        # 解码器权重
        self.W_decoder_h1 = np.random.randn(latent_dim, hidden_dim) / np.sqrt(latent_dim)
        self.b_decoder_h1 = np.zeros((1, hidden_dim))
        self.W_decoder_h2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b_decoder_h2 = np.zeros((1, hidden_dim))
        self.W_decoder = np.random.randn(hidden_dim, input_dim) / np.sqrt(hidden_dim)
        self.b_decoder = np.zeros((1, input_dim))

    def _relu(self, Z):
        """ReLU激活函数，带有数值稳定性处理"""
        return np.maximum(0, np.minimum(Z, 20))  # 限制最大值以提高稳定性

    def _sigmoid(self, Z):
        """Sigmoid激活函数，带有数值稳定性处理"""
        Z = np.clip(Z, -20, 20)  # 限制输入范围以提高数值稳定性
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
        """编码器：增加两个隐藏层"""
        # 第一隐藏层
        h1 = self._relu(np.dot(X, self.W_encoder_h1) + self.b_encoder_h1)
        # 第二隐藏层
        h2 = self._relu(np.dot(h1, self.W_encoder_h2) + self.b_encoder_h2)
        # 输出层
        Z_encoded = np.dot(h2, self.W_encoder) + self.b_encoder
        mean = Z_encoded[:, :self.latent_dim]
        log_var = Z_encoded[:, self.latent_dim:]
        return mean, log_var

    def _decode(self, z):
        """解码器：增加两个隐藏层"""
        # 第一隐藏层
        h1 = self._relu(np.dot(z, self.W_decoder_h1) + self.b_decoder_h1)
        # 第二隐藏层
        h2 = self._relu(np.dot(h1, self.W_decoder_h2) + self.b_decoder_h2)
        # 输出层
        decoded = self._sigmoid(np.dot(h2, self.W_decoder) + self.b_decoder)
        return decoded

    def _compute_loss(self, X, decoded, mean, log_var):
        """
        计算VAE的损失函数，包括重构误差和KL散度。
        添加了权重因子来平衡两个损失项。
        """
        # 重构误差（二元交叉熵）
        reconstruction_loss = -np.mean(X * np.log(decoded + 1e-10) + (1 - X) * np.log(1 - decoded + 1e-10))

        # KL散度
        kl_divergence = -0.5 * np.mean(1 + log_var - np.square(mean) - np.exp(log_var))

        # 总损失（添加权重因子beta来控制KL项的影响）
        beta = 0.5  # 增加beta值以平衡KL项
        loss = reconstruction_loss + beta * kl_divergence
        return loss, reconstruction_loss, kl_divergence

    def _backward(self, X, decoded, mean, log_var, z):
        """反向传播：计算梯度并更新权重和偏置"""
        m = X.shape[0]

        # 解码器的反向传播
        dZ_decoded = -(X / (decoded + 1e-10) - (1 - X) / (1 - decoded + 1e-10))
        dZ_decoded = dZ_decoded * (decoded * (1 - decoded))  # sigmoid导数

        # 解码器隐藏层
        dW_decoder = np.dot(self._relu(np.dot(z, self.W_decoder_h1) + self.b_decoder_h1).T, dZ_decoded) / m
        db_decoder = np.sum(dZ_decoded, axis=0, keepdims=True) / m

        dh_decoder = np.dot(dZ_decoded, self.W_decoder.T)
        dh_decoder = dh_decoder * (np.dot(z, self.W_decoder_h1) + self.b_decoder_h1 > 0)  # ReLU导数

        dW_decoder_h1 = np.dot(z.T, dh_decoder) / m
        db_decoder_h1 = np.sum(dh_decoder, axis=0, keepdims=True) / m

        # 编码器的反向传播
        dmean = mean  # KL散度对均值的梯度
        dlog_var = 0.5 * (np.exp(log_var) - 1)  # KL散度对对数方差的梯度

        # 梯度裁剪
        clip_value = 5.0
        dmean = np.clip(dmean, -clip_value, clip_value)
        dlog_var = np.clip(dlog_var, -clip_value, clip_value)

        # 编码器隐藏层
        dZ_encoder = np.concatenate([dmean, dlog_var], axis=1)
        h_encoder = self._relu(np.dot(X, self.W_encoder_h1) + self.b_encoder_h1)

        dW_encoder = np.dot(h_encoder.T, dZ_encoder) / m
        db_encoder = np.sum(dZ_encoder, axis=0, keepdims=True) / m

        dh_encoder = np.dot(dZ_encoder, self.W_encoder.T)
        dh_encoder = dh_encoder * (np.dot(X, self.W_encoder_h1) + self.b_encoder_h1 > 0)

        dW_encoder_h1 = np.dot(X.T, dh_encoder) / m
        db_encoder_h1 = np.sum(dh_encoder, axis=0, keepdims=True) / m

        # 更新权重和偏置（添加动量）
        momentum = 0.9
        if not hasattr(self, 'v_W_encoder'):
            self.v_W_encoder = 0
            self.v_b_encoder = 0
            self.v_W_encoder_h1 = 0
            self.v_b_encoder_h1 = 0
            self.v_W_decoder = 0
            self.v_b_decoder = 0
            self.v_W_decoder_h1 = 0
            self.v_b_decoder_h1 = 0

        # 编码器更新
        self.v_W_encoder = momentum * self.v_W_encoder - self.learning_rate * dW_encoder
        self.v_b_encoder = momentum * self.v_b_encoder - self.learning_rate * db_encoder
        self.v_W_encoder_h1 = momentum * self.v_W_encoder_h1 - self.learning_rate * dW_encoder_h1
        self.v_b_encoder_h1 = momentum * self.v_b_encoder_h1 - self.learning_rate * db_encoder_h1

        self.W_encoder += self.v_W_encoder
        self.b_encoder += self.v_b_encoder
        self.W_encoder_h1 += self.v_W_encoder_h1
        self.b_encoder_h1 += self.v_b_encoder_h1

        # 解码器更新
        self.v_W_decoder = momentum * self.v_W_decoder - self.learning_rate * dW_decoder
        self.v_b_decoder = momentum * self.v_b_decoder - self.learning_rate * db_decoder
        self.v_W_decoder_h1 = momentum * self.v_W_decoder_h1 - self.learning_rate * dW_decoder_h1
        self.v_b_decoder_h1 = momentum * self.v_b_decoder_h1 - self.learning_rate * db_decoder_h1

        self.W_decoder += self.v_W_decoder
        self.b_decoder += self.v_b_decoder
        self.W_decoder_h1 += self.v_W_decoder_h1
        self.b_decoder_h1 += self.v_b_decoder_h1

    def fit(self, X):
        """训练变分自编码器（VAE）模型"""
        # 数据预处理：归一化到[0,1]区间
        X = (X - X.min()) / (X.max() - X.min())

        best_loss = float('inf')
        patience = 100000  # 早停的耐心值
        no_improve = 0

        # 用于记录每个epoch的损失
        losses = []

        for epoch in range(self.max_iter):
            # 编码器：得到均值和对数方差
            mean, log_var = self._encode(X)

            # 从均值和对数方差中采样得到潜在变量
            z = self._sampling(mean, log_var)

            # 解码器：从潜在空间采样生成重构数据
            decoded = self._decode(z)

            # 计算损失
            loss, reconstruction_loss, kl_divergence = self._compute_loss(X, decoded, mean, log_var)

            # 记录损失
            losses.append(loss)

            # 早停检查
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n提前停止训练，在epoch {epoch + 1}处")
                    break

            # 反向传播：更新权重和偏置
            self._backward(X, decoded, mean, log_var, z)

            # 每100个epoch输出一次损失
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss:.4f}, Reconstruction Loss: {reconstruction_loss:.4f}, KL Divergence: {kl_divergence:.4f}")

        # 绘制损失曲线图
        plt.plot(losses)
        plt.title('VAE Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, X):
        """使用训练好的VAE进行数据重构"""
        # 数据预处理：归一化到[0,1]区间
        X = (X - X.min()) / (X.max() - X.min())

        mean, log_var = self._encode(X)
        z = self._sampling(mean, log_var)
        reconstructed = self._decode(z)

        # 还原到原始数据范围
        reconstructed = reconstructed * (X.max() - X.min()) + X.min()
        return reconstructed


# 示例：创建VAE模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据
    np.random.seed(42)  # 设置随机种子以保证结果可重现
    X = np.random.randn(1000, 10)  # 1000个样本，10维输入

    # 创建VAE模型（调整超参数）
    vae = VAE(input_dim=10, latent_dim=4, learning_rate=0.0001, max_iter=3000)

    # 训练VAE模型
    vae.fit(X)

    # 使用训练好的VAE进行数据重构
    print("\n测试重构效果：")
    test_samples = X[:5]  # 取前5个样本进行测试
    print("原始数据：")
    print(test_samples)
    reconstructed = vae.predict(test_samples)
    print("\n重构数据：")
    print(reconstructed)

    # 计算重构误差
    mse = np.mean((test_samples - reconstructed) ** 2)
    print(f"\n平均重构误差（MSE）：{mse:.4f}")

    # 生成新样本
    print("\n从潜在空间采样生成新数据：")
    z_new = np.random.randn(5, 4)  # 从标准正态分布采样
    generated = vae._decode(z_new)
    print(generated)
