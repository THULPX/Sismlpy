import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- 生成对抗网络（GAN）算法 -----------------------------

# 介绍：
# 生成对抗网络（GAN）由两部分组成：生成器（Generator）和判别器（Discriminator）。
# 生成器负责生成伪造的样本，判别器则用于判断样本是真实的还是由生成器伪造的。
# 两者通过对抗训练的方式相互竞争，最终生成器能够生成越来越真实的样本。
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
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # 使用He初始化
        self.W1 = np.random.randn(latent_dim, 32) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, output_dim) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, output_dim))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def forward(self, z):
        self.z = z
        self.a1 = np.dot(z, self.W1) + self.b1
        self.h1 = self.leaky_relu(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.out = np.tanh(self.a2)
        return self.out

    def backward(self, grad_output):
        m = self.z.shape[0]

        # 输出层梯度（tanh导数）
        grad_a2 = grad_output * (1 - np.square(self.out))

        # 隐藏层梯度
        grad_h1 = np.dot(grad_a2, self.W2.T)
        grad_a1 = grad_h1 * self.leaky_relu_derivative(self.a1)

        # 参数梯度
        grad_W2 = np.dot(self.h1.T, grad_a2) / m
        grad_b2 = np.mean(grad_a2, axis=0, keepdims=True)
        grad_W1 = np.dot(self.z.T, grad_a1) / m
        grad_b1 = np.mean(grad_a1, axis=0, keepdims=True)

        # 更新参数
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1


class Discriminator:
    def __init__(self, input_dim, learning_rate=0.0002):
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # 使用He初始化
        self.W1 = np.random.randn(input_dim, 32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, 1))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x):
        self.x = x
        self.a1 = np.dot(x, self.W1) + self.b1
        self.h1 = self.leaky_relu(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.out = self.sigmoid(self.a2)
        return self.out

    def backward(self, grad_output):
        m = self.x.shape[0]

        # 输出层梯度（sigmoid导数）
        grad_a2 = grad_output * self.out * (1 - self.out)

        # 隐藏层梯度
        grad_h1 = np.dot(grad_a2, self.W2.T)
        grad_a1 = grad_h1 * self.leaky_relu_derivative(self.a1)

        # 参数梯度
        grad_W2 = np.dot(self.h1.T, grad_a2) / m
        grad_b2 = np.mean(grad_a2, axis=0, keepdims=True)
        grad_W1 = np.dot(self.x.T, grad_a1) / m
        grad_b1 = np.mean(grad_a1, axis=0, keepdims=True)

        # 更新参数
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1


class GAN:
    def __init__(self, latent_dim, input_dim):
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # 使用不同的学习率
        self.generator = Generator(latent_dim, input_dim, learning_rate=0.0002)
        self.discriminator = Discriminator(input_dim, learning_rate=0.0002)  # 增加判别器的学习率

        self.d_losses = []
        self.g_losses = []

    def train_discriminator(self, X_real, X_fake):
        # 真实样本的预测
        d_real = self.discriminator.forward(X_real)
        d_real_loss = -np.mean(np.log(d_real + 1e-8))
        d_real_grad = -1.0 / (d_real + 1e-8)
        self.discriminator.backward(d_real_grad)

        # 生成样本的预测
        d_fake = self.discriminator.forward(X_fake)
        d_fake_loss = -np.mean(np.log(1 - d_fake + 1e-8))
        d_fake_grad = 1.0 / (1 - d_fake + 1e-8)
        self.discriminator.backward(d_fake_grad)

        return d_real_loss + d_fake_loss

    def train_generator(self, z):
        # 生成假样本
        X_fake = self.generator.forward(z)
        d_fake = self.discriminator.forward(X_fake)

        # 计算生成器损失
        g_loss = -np.mean(np.log(d_fake + 1e-8))

        # 计算梯度并更新生成器
        g_grad = -1.0 / (d_fake + 1e-8)
        self.generator.backward(g_grad)

        return g_loss

    def train_step(self, X_real, batch_size):
        # 生成随机噪声（进一步增加噪声多样性）
        z = np.random.normal(0, 1.0, (batch_size, self.latent_dim))

        # 生成假样本
        X_fake = self.generator.forward(z)

        # 训练判别器
        d_loss = self.train_discriminator(X_real, X_fake)

        # 训练生成器（增加生成器训练频次）
        g_loss = 0
        for _ in range(3):  # 增加生成器更新频次
            z = np.random.normal(0, 1.0, (batch_size, self.latent_dim))
            g_loss += self.train_generator(z)
        g_loss /= 3

        return d_loss, g_loss

    def fit(self, X_real, epochs=5000, batch_size=32):
        n_batches = len(X_real) // batch_size

        for epoch in range(epochs):
            # 打乱数据
            np.random.shuffle(X_real)

            d_epoch_loss = 0
            g_epoch_loss = 0

            for i in range(n_batches):
                batch_real = X_real[i * batch_size:(i + 1) * batch_size]
                d_loss, g_loss = self.train_step(batch_real, batch_size)

                d_epoch_loss += d_loss
                g_epoch_loss += g_loss

            d_epoch_loss /= n_batches
            g_epoch_loss /= n_batches

            self.d_losses.append(d_epoch_loss)
            self.g_losses.append(g_epoch_loss)

            if epoch % 500 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_epoch_loss:.4f}, G Loss: {g_epoch_loss:.4f}")

    def generate(self, n_samples):
        z = np.random.normal(0, 1.0, (n_samples, self.latent_dim))
        return self.generator.forward(z)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator')
        plt.plot(self.g_losses, label='Generator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # 生成训练数据
    n_samples = 1000
    mean = [2, 3]
    cov = [[2.0, 0.5], [0.5, 2.0]]  # 增加方差和添加相关性
    X_real = np.random.multivariate_normal(mean, cov, n_samples)

    # 数据标准化
    X_mean = np.mean(X_real, axis=0)
    X_std = np.std(X_real, axis=0)
    X_real = (X_real - X_mean) / X_std

    # 创建和训练GAN
    gan = GAN(latent_dim=2, input_dim=2)
    gan.fit(X_real, epochs=5000, batch_size=32)

    # 生成样本并还原到原始尺度
    fake_samples = gan.generate(10)  # 生成更多样本
    fake_samples = fake_samples * X_std + X_mean

    print("\n生成的伪造样本：")
    print(fake_samples)

    # 绘制真实数据和生成数据的散点图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_real[:, 0] * X_std[0] + X_mean[0],
                X_real[:, 1] * X_std[1] + X_mean[1],
                alpha=0.5, label='Real')
    plt.title('Real Data Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    fake_data = gan.generate(1000)
    fake_data = fake_data * X_std + X_mean
    plt.scatter(fake_data[:, 0], fake_data[:, 1],
                alpha=0.5, label='Generated')
    plt.title('Generated Data Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制损失
    gan.plot_losses()
