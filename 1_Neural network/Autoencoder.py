import numpy as np
import matplotlib.pyplot as plt


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
        
        # 定义网络结构（使用更深的网络）
        self.hidden_dims = [32, 16, latent_dim, 16, 32]  # 对称的网络结构
        
        # 初始化所有层的权重和偏置
        self.weights = []
        self.biases = []
        self.velocities_w = []  # 动量
        self.velocities_b = []
        self.momentum = 0.9
        
        # 编码器权重
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims[:3]:  # 前半部分是编码器
            W = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros((1, hidden_dim))
            self.weights.append(W)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(W))
            self.velocities_b.append(np.zeros_like(b))
            prev_dim = hidden_dim
            
        # 解码器权重
        for hidden_dim in self.hidden_dims[3:]:  # 后半部分是解码器
            W = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros((1, hidden_dim))
            self.weights.append(W)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(W))
            self.velocities_b.append(np.zeros_like(b))
            prev_dim = hidden_dim
            
        # 输出层
        W = np.random.randn(prev_dim, input_dim) * np.sqrt(2.0 / prev_dim)
        b = np.zeros((1, input_dim))
        self.weights.append(W)
        self.biases.append(b)
        self.velocities_w.append(np.zeros_like(W))
        self.velocities_b.append(np.zeros_like(b))

    def _leaky_relu(self, Z, alpha=0.01):
        """LeakyReLU激活函数"""
        return np.where(Z > 0, Z, alpha * Z)
    
    def _leaky_relu_derivative(self, Z, alpha=0.01):
        """LeakyReLU导数"""
        return np.where(Z > 0, 1, alpha)

    def _forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        # 前向传播通过所有层
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # 最后一层使用线性激活，其他层使用LeakyReLU
            if i == len(self.weights) - 1:
                current_input = z
            else:
                current_input = self._leaky_relu(z)
            self.activations.append(current_input)
            
        return current_input

    def _backward(self, X, output, learning_rate):
        """反向传播"""
        m = X.shape[0]
        delta = (output - X) / m  # 初始误差
        
        # 从最后一层开始反向传播
        for i in range(len(self.weights)-1, -1, -1):
            # 计算梯度
            dW = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # 应用动量
            self.velocities_w[i] = self.momentum * self.velocities_w[i] - learning_rate * dW
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - learning_rate * db
            
            # 更新权重和偏置
            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]
            
            # 计算下一层的delta（如果不是第一层）
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._leaky_relu_derivative(self.z_values[i-1])

    def fit(self, X, batch_size=32):
        """训练自编码器模型"""
        # 数据归一化
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X = (X - self.mean) / self.std
        
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience = 50
        no_improve = 0
        
        # 学习率衰减
        initial_lr = self.learning_rate
        decay_rate = 0.95
        decay_steps = 100
        
        # 记录训练过程中的loss
        self.loss_history = []
        
        for epoch in range(self.max_iter):
            # 学习率衰减
            current_lr = initial_lr * (decay_rate ** (epoch // decay_steps))
            
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            total_loss = 0
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                batch_X = X[batch_indices]
                
                # 前向传播
                output = self._forward(batch_X)
                
                # 计算损失
                loss = np.mean((batch_X - output) ** 2)
                total_loss += loss * len(batch_indices)
                
                # 反向传播
                self._backward(batch_X, output, current_lr)
            
            # 计算平均损失
            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"\n提前停止训练于epoch {epoch+1}，最佳loss: {best_loss:.6f}")
                break

            # 每100个epoch输出一次损失
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    def predict(self, X):
        """使用训练好的自编码器进行数据重构"""
        # 对输入数据进行归一化
        X = (X - self.mean) / self.std
        
        # 获取重构结果
        output = self._forward(X)
        
        # 反归一化输出
        return output * self.std + self.mean


# 示例：创建自编码器模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    X = np.random.randn(1000, 3)  # 1000个样本
    
    # 创建自编码器模型
    autoencoder = Autoencoder(input_dim=3, latent_dim=2, learning_rate=0.005, max_iter=1500)

    # 训练自编码器
    autoencoder.fit(X, batch_size=32)

    # 使用训练好的自编码器进行数据重构
    X_reconstructed = autoencoder.predict(X)
    
    # 计算重构误差
    reconstruction_error = np.mean(np.square(X - X_reconstructed))
    print(f"\n平均重构误差: {reconstruction_error:.6f}")
    print("\n原始数据样本：")
    print(X[:3])
    print("\n重构数据样本：")
    print(X_reconstructed[:3])
    
    # 计算每个维度的重构误差
    dim_errors = np.mean(np.square(X - X_reconstructed), axis=0)
    print("\n各维度重构误差：")
    for i, error in enumerate(dim_errors):
        print(f"维度 {i}: {error:.6f}")
        
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.loss_history, 'b-', label='Training Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # 使用对数尺度更好地显示loss的变化
    plt.savefig('autoencoder_loss.png')
    plt.close()
