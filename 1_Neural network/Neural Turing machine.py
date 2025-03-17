import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ----------------------------- 神经图灵机（NTM）算法 -----------------------------

# 介绍：
# 神经图灵机（Neural Turing Machine, NTM）是一种结合了神经网络和图灵机的模型，旨在模拟图灵机的存储和操作能力。
# NTM由一个神经网络控制器和一个外部存储器（类似图灵机的带）组成，控制器通过对存储器进行读写操作来处理输入数据。
# NTM能够进行复杂的任务，如序列生成、排序、复制等。

# 输入输出：
# 输入：
# - X: 输入数据，形状为 (n_samples, input_dim)。
# 输出：
# - 模型的输出，可能是预测值或其他任务的结果。

# 算法步骤：
# 1. 控制器（通常是RNN）接收输入并生成一个指令，用于访问外部存储器（读/写/擦除等）。
# 2. 存储器包含一个带状结构，控制器根据指令进行读写操作。
# 3. 控制器输出结果并决定接下来的操作。
# 4. 重复以上步骤直至任务完成。
# 5. NTM通过反向传播来优化控制器和存储器的参数。

# 主要参数：
# - input_dim: 输入数据的维度。
# - memory_size: 存储器的大小。
# - memory_dim: 存储器每个单元的维度。
# - latent_dim: 控制器的维度。
# - learning_rate: 学习率。
# - max_iter: 最大训练迭代次数。


class NTM:
    def __init__(self, input_dim, memory_size, memory_dim, latent_dim, learning_rate=0.0001, max_iter=1000):
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # 数据标准化
        self.scaler = StandardScaler()

        # 初始化权重（使用He初始化）
        def he_init(size):
            fan_in = size[0]
            limit = np.sqrt(2.0 / fan_in)
            return np.random.normal(0, limit, size)

        # 控制器权重
        controller_input_dim = input_dim + memory_size * memory_dim
        self.W_controller = he_init((controller_input_dim, latent_dim))  # 简化了模型
        self.b_controller = np.zeros((1, latent_dim))

        # 存储器
        self.memory = he_init((memory_size, memory_dim))

        # 输出层权重
        self.W_output = he_init((memory_dim, input_dim))
        self.b_output = np.zeros((1, input_dim))

        # 优化器状态
        self.m = {}
        self.v = {}
        for param_name in ['W_controller', 'b_controller', 'W_output', 'b_output']:
            self.m[param_name] = np.zeros_like(getattr(self, param_name))
            self.v[param_name] = np.zeros_like(getattr(self, param_name))

        # 记录训练历史
        self.history = {'loss': [], 'val_loss': []}

    def _sigmoid(self, Z):
        """Sigmoid激活函数（带数值稳定性）"""
        Z = np.clip(Z, -100, 100)
        return 1.0 / (1.0 + np.exp(-Z))

    def _tanh(self, Z):
        """Tanh激活函数（带数值稳定性）"""
        return np.clip(np.tanh(Z), -1 + 1e-7, 1 - 1e-7)

    def _softmax(self, Z):
        """Softmax激活函数（带数值稳定性）"""
        Z = np.clip(Z, -100, 100)
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return exp_Z / (np.sum(exp_Z, axis=-1, keepdims=True) + 1e-7)

    def _controller_step(self, input_data, prev_memory):
        """控制器前向传播"""
        input_data = np.array(input_data).reshape(1, -1)
        prev_memory_flat = prev_memory.reshape(1, -1)
        controller_input = np.concatenate([input_data, prev_memory_flat], axis=1)
        hidden = np.dot(controller_input, self.W_controller) + self.b_controller
        hidden = self._tanh(hidden)

        # 修改read_weights和write_weights的计算，确保维度匹配
        read_logits = np.dot(hidden, np.random.randn(self.latent_dim, self.memory_size))
        write_logits = np.dot(hidden, np.random.randn(self.latent_dim, self.memory_size))
        read_weights = self._softmax(read_logits)
        write_weights = self._softmax(write_logits)
        write_data = self._tanh(np.dot(hidden, np.random.randn(self.latent_dim, self.memory_dim)))
        
        return hidden, read_weights.flatten(), write_weights.flatten(), write_data.reshape(-1)

    def _read_memory(self, read_weights):
        """从存储器中读取信息"""
        read_weights = read_weights.reshape(1, -1)[:, :self.memory_size]  # 确保维度匹配
        context = np.dot(read_weights, self.memory)  # 修复矩阵乘法
        output = np.dot(context, self.W_output) + self.b_output
        return self._tanh(output).flatten()

    def _write_memory(self, write_weights, write_data):
        """将数据写入存储器"""
        write_weights = write_weights.reshape(-1, 1)  # 调整形状为(memory_size, 1)
        write_data = write_data.reshape(1, -1)[:, :self.memory_dim]  # 调整形状为(1, memory_dim)
        self.memory = self.memory * (1 - write_weights) + write_weights * write_data
        self.memory = np.clip(self.memory, -1, 1)

    def _adam_update(self, param_name, grad, t):
        """Adam优化器更新"""
        m = self.m[param_name]
        v = self.v[param_name]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        self.m[param_name] = m
        self.v[param_name] = v
        return -self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def fit(self, X, validation_split=0.2):
        """训练模型（带学习率调度和改进的损失函数）"""
        X = self.scaler.fit_transform(X)
        val_size = int(len(X) * validation_split)
        train_data = X[:-val_size]
        val_data = X[-val_size:]
        prev_memory = np.zeros((self.memory_size, self.memory_dim))

        best_val_loss = float('inf')
        patience = 50
        no_improve_count = 0
        initial_lr = self.learning_rate

        for epoch in range(self.max_iter):
            self.learning_rate = initial_lr * (1 + np.cos(epoch * np.pi / self.max_iter)) / 2
            total_loss = 0
            for t, input_data in enumerate(train_data, 1):
                controller_output, read_weights, write_weights, write_data = self._controller_step(input_data,
                                                                                                   prev_memory)
                read_data = self._read_memory(read_weights)
                self._write_memory(write_weights, write_data)
                error = input_data - read_data
                loss = np.mean(error ** 2)  # MSE损失
                total_loss += loss

                # 更新参数
                for param_name in self.m.keys():
                    param = getattr(self, param_name)
                    grad = np.random.randn(*param.shape) * 0.01  # 模拟梯度，需实际计算
                    update = self._adam_update(param_name, grad, t)
                    setattr(self, param_name, param + update)

                prev_memory = np.clip(self.memory.copy(), -1, 1)

            avg_train_loss = total_loss / len(train_data)
            self.history['loss'].append(avg_train_loss)

            # 验证
            val_loss = 0
            for val_input in val_data:
                controller_output, read_weights, write_weights, write_data = self._controller_step(val_input,
                                                                                                   prev_memory)
                read_data = self._read_memory(read_weights)
                val_loss += np.mean((val_input - read_data) ** 2)

            avg_val_loss = val_loss / len(val_data)
            self.history['val_loss'].append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X):
        """模型预测"""
        X = self.scaler.transform(X)
        predictions = []
        prev_memory = np.zeros((self.memory_size, self.memory_dim))
        for input_data in X:
            controller_output, read_weights, write_weights, write_data = self._controller_step(input_data, prev_memory)
            read_data = self._read_memory(read_weights)
            self._write_memory(write_weights, write_data)
            predictions.append(read_data)
            prev_memory = self.memory.copy()
        predictions = np.array(predictions)
        return self.scaler.inverse_transform(predictions)

    def plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


# 示例：创建NTM模型并训练
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1000, 10)

    # 创建并训练模型
    model = NTM(
        input_dim=10,
        memory_size=64,  # 存储器大小
        memory_dim=8,  # 存储器维度
        latent_dim=16,  # 控制器维度
        learning_rate=0.0001,
        max_iter=100
    )

    # 训练模型
    model.fit(X)

    # 绘制训练历史
    model.plot_history()

    # 预测
    X_test = np.random.randn(10, 10)
    predictions = model.predict(X_test)
    print("预测结果:", predictions.shape)
