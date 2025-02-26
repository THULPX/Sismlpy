import numpy as np


# ----------------------------- 生成对抗网络（GAN）算法 -----------------------------

# 介绍：
# 生成对抗网络（GAN）由两部分组成：生成器（Generator）和判别器（Discriminator）。生成器负责生成伪造的样本，判别器
# 则用于判断样本是真实的还是由生成器伪造的。两者通过对抗训练的方式相互竞争，最终生成器能够生成越来越真实的样本。
# GAN广泛应用于图像生成、图像超分辨率、文本生成等领域。

# 输入输出：
# 输入：
# - X: 训练数据，形状为 (n_samples, input_size)。
# 输出：
# - 生成器和判别器的训练好的权重和偏置。

# 算法步骤：
# 1. 初始化生成器和判别器的网络参数。
# 2. 训练判别器：使用真实样本和生成器生成的假样本来训练判别器。
# 3. 训练生成器：通过训练生成器使其能够生成更真实的样本，从而“欺骗”判别器。
# 4. 重复步骤2和步骤3，直到生成器生成的样本能够骗过判别器。

# 主要参数：
# - learning_rate：学习率，用于更新权重。
# - max_iter：最大迭代次数。
# - latent_dim：生成器输入的潜在空间的维度。
# - output_dim：生成器输出的样本维度（与训练数据一致）。

class Generator:
    def __init__(self, latent_dim, output_dim, learning_rate=0.0002):
        """
        初始化生成器模型。

        :param latent_dim: 生成器输入的潜在空间的维度。
        :param output_dim: 生成器输出的样本维度（与训练数据一致）。
        :param learning_rate: 学习率。
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # 初始化生成器的权重和偏置
        self.W1 = np.random.randn(latent_dim, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def _relu(self, Z):
        """
        ReLU 激活函数。

        :param Z: 输入数据。
        :return: ReLU 激活后的输出。
        """
        return np.maximum(0, Z)

    def _tanh(self, Z):
        """
        Tanh 激活函数。

        :param Z: 输入数据。
        :return: Tanh 激活后的输出。
        """
        return np.tanh(Z)

    def _forward(self, z):
        """
        前向传播：生成器生成伪造样本。

        :param z: 潜在空间输入数据。
        :return: 生成的伪造样本。
        """
        hidden = self._relu(np.dot(z, self.W1) + self.b1)
        output = self._tanh(np.dot(hidden, self.W2) + self.b2)
        return output

    def get_parameters(self):
        """
        获取生成器的所有参数（用于反向传播）。

        :return: 生成器的权重和偏置。
        """
        return [self.W1, self.b1, self.W2, self.b2]


class Discriminator:
    def __init__(self, input_dim, learning_rate=0.0002):
        """
        初始化判别器模型。

        :param input_dim: 输入数据的维度。
        :param learning_rate: 学习率。
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # 初始化判别器的权重和偏置
        self.W1 = np.random.randn(input_dim, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def _sigmoid(self, Z):
        """
        Sigmoid 激活函数。

        :param Z: 输入数据。
        :return: Sigmoid 激活后的输出。
        """
        return 1 / (1 + np.exp(-Z))

    def _forward(self, x):
        """
        前向传播：判别器判断样本是否真实。

        :param x: 输入数据（样本）。
        :return: 判别器的输出（0表示假，1表示真）。
        """
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU激活
        output = self._sigmoid(np.dot(hidden, self.W2) + self.b2)  # Sigmoid激活
        return output

    def get_parameters(self):
        """
        获取判别器的所有参数（用于反向传播）。

        :return: 判别器的权重和偏置。
        """
        return [self.W1, self.b1, self.W2, self.b2]


class GAN:
    def __init__(self, latent_dim, input_dim, learning_rate=0.0002, max_iter=10000):
        """
        初始化生成对抗网络（GAN）。

        :param latent_dim: 生成器输入的潜在空间的维度。
        :param input_dim: 训练数据的输入维度。
        :param learning_rate: 学习率。
        :param max_iter: 最大迭代次数。
        """
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # 创建生成器和判别器
        self.generator = Generator(latent_dim, input_dim, learning_rate)
        self.discriminator = Discriminator(input_dim, learning_rate)

    def _train_discriminator(self, X_real, X_fake):
        """
        训练判别器：使用真实样本和生成器生成的假样本来训练判别器。

        :param X_real: 真实样本。
        :param X_fake: 生成器生成的假样本。
        """
        m_real = X_real.shape[0]
        m_fake = X_fake.shape[0]

        # 真实样本的标签为1
        real_labels = np.ones((m_real, 1))
        # 假样本的标签为0
        fake_labels = np.zeros((m_fake, 1))

        # 计算判别器对真实样本的损失
        real_output = self.discriminator._forward(X_real)
        real_loss = np.mean((real_output - real_labels) ** 2)

        # 计算判别器对假样本的损失
        fake_output = self.discriminator._forward(X_fake)
        fake_loss = np.mean((fake_output - fake_labels) ** 2)

        # 反向传播（梯度下降）
        total_loss = real_loss + fake_loss
        return total_loss

    def _train_generator(self, X_fake):
        """
        训练生成器：通过训练生成器使其能够生成更真实的样本，欺骗判别器。

        :param X_fake: 生成器生成的假样本。
        """
        # 生成器目标是让判别器误判假样本为真实样本
        labels = np.ones((X_fake.shape[0], 1))
        output = self.discriminator._forward(X_fake)
        loss = np.mean((output - labels) ** 2)
        return loss

    def fit(self, X_real):
        """
        训练生成对抗网络（GAN）。

        :param X_real: 真实样本数据。
        """
        for epoch in range(self.max_iter):
            # 训练判别器
            z = np.random.randn(X_real.shape[0], self.latent_dim)  # 潜在空间输入
            X_fake = self.generator._forward(z)  # 生成器生成假样本
            d_loss = self._train_discriminator(X_real, X_fake)

            # 训练生成器
            z = np.random.randn(X_real.shape[0], self.latent_dim)  # 生成器潜在输入
            X_fake = self.generator._forward(z)  # 生成假样本
            g_loss = self._train_generator(X_fake)

            # 输出损失
            print(f"Epoch {epoch + 1}/{self.max_iter}, D Loss: {d_loss}, G Loss: {g_loss}")

    def generate(self, n_samples):
        """
        生成伪造样本。

        :param n_samples: 生成的伪造样本数量。
        :return: 生成的伪造样本。
        """
        z = np.random.randn(n_samples, self.latent_dim)  # 随机潜在空间输入
        return self.generator._forward(z)


# 示例：创建GAN模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据（如二维正态分布样本）
    X_real = np.random.randn(1000, 2)

    # 创建GAN模型并训练
    gan = GAN(latent_dim=2, input_dim=2, learning_rate=0.0002, max_iter=10000)
    gan.fit(X_real)

    # 生成一些伪造样本
    fake_samples = gan.generate(5)
    print("生成的伪造样本：", fake_samples)
