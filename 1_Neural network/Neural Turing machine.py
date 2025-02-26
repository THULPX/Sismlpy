import numpy as np


# ----------------------------- 神经图灵机（NTM）算法 -----------------------------

# 介绍：
# 神经图灵机（Neural Turing Machine, NTM）是一种结合了神经网络和图灵机的模型，旨在模拟图灵机的存储和操作能力。
# NTM由一个神经网络控制器和一个外部存储器（类似图灵机的带）组成，控制器通过对存储器进行读写操作来处理输入数据。NTM能够进行复杂的任务，如序列生成、排序、复制等。

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
    def __init__(self, input_dim, memory_size, memory_dim, latent_dim, learning_rate=0.001, max_iter=1000):
        """
        初始化神经图灵机（NTM）模型。

        :param input_dim: 输入数据的维度。
        :param memory_size: 存储器的大小（存储单元数量）。
        :param memory_dim: 存储器每个单元的维度。
        :param latent_dim: 控制器的维度。
        :param learning_rate: 学习率。
        :param max_iter: 最大训练迭代次数。
        """
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # 初始化控制器（使用LSTM作为控制器）
        self.W_controller = np.random.randn(input_dim + memory_dim, latent_dim) * 0.01
        self.b_controller = np.zeros((1, latent_dim))

        # 初始化存储器
        self.memory = np.random.randn(memory_size, memory_dim) * 0.01

        # 初始化控制器输出的操作（读写头）
        self.W_read = np.random.randn(latent_dim, memory_size) * 0.01
        self.W_write = np.random.randn(latent_dim, memory_size) * 0.01
        self.b_read = np.zeros((1, memory_size))
        self.b_write = np.zeros((1, memory_size))

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

    def _read_memory(self, read_weights):
        """
        从存储器中读取信息。

        :param read_weights: 读权重。
        :return: 从存储器读取的内容。
        """
        return np.dot(read_weights, self.memory)

    def _write_memory(self, write_weights, write_data):
        """
        将数据写入存储器。

        :param write_weights: 写权重。
        :param write_data: 写入的数据。
        """
        self.memory += np.outer(write_weights, write_data)

    def _controller_step(self, input_data, prev_memory, prev_read_weights):
        """
        控制器（LSTM）进行一步的计算，并生成读/写操作。

        :param input_data: 输入数据。
        :param prev_memory: 上一时刻的存储器内容。
        :param prev_read_weights: 上一时刻的读权重。
        :return: 控制器输出、读权重、写权重、写数据。
        """
        # 拼接输入数据和上一时刻的记忆内容，送入控制器
        controller_input = np.concatenate([input_data, prev_memory])

        # 控制器的输出
        controller_output = self._tanh(np.dot(controller_input, self.W_controller) + self.b_controller)

        # 读权重和写权重（使用sigmoid激活）
        read_weights = self._sigmoid(np.dot(controller_output, self.W_read) + self.b_read)
        write_weights = self._sigmoid(np.dot(controller_output, self.W_write) + self.b_write)

        # 写数据（随机初始化）
        write_data = np.random.randn(self.memory_dim)

        return controller_output, read_weights, write_weights, write_data

    def fit(self, X):
        """
        训练神经图灵机（NTM）模型。

        :param X: 输入数据。
        """
        prev_memory = np.zeros((self.memory_size, self.memory_dim))
        prev_read_weights = np.zeros((self.memory_size,))

        for epoch in range(self.max_iter):
            loss = 0
            for input_data in X:
                # 控制器计算步骤
                controller_output, read_weights, write_weights, write_data = self._controller_step(
                    input_data, prev_memory, prev_read_weights
                )

                # 从存储器中读取数据
                read_data = self._read_memory(read_weights)

                # 将数据写入存储器
                self._write_memory(write_weights, write_data)

                # 计算损失（简单的重构误差）
                loss += np.sum((input_data - read_data) ** 2)

                # 更新上一个存储器状态和读权重
                prev_memory = self.memory
                prev_read_weights = read_weights

            # 输出每个epoch的损失
            print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss}")

    def predict(self, X):
        """
        使用训练好的NTM进行数据预测。

        :param X: 输入数据。
        :return: NTM的输出。
        """
        prev_memory = np.zeros((self.memory_size, self.memory_dim))
        prev_read_weights = np.zeros((self.memory_size,))
        predictions = []

        for input_data in X:
            controller_output, read_weights, _, _ = self._controller_step(input_data, prev_memory, prev_read_weights)

            # 从存储器中读取数据
            read_data = self._read_memory(read_weights)

            predictions.append(read_data)
            prev_memory = self.memory
            prev_read_weights = read_weights

        return np.array(predictions)


# 示例：创建NTM模型并训练
if __name__ == "__main__":
    # 生成一个简单的训练数据（例如，二维数据）
    X = np.random.randn(100, 3)  # 100个样本，3维输入

    # 创建NTM模型
    ntm = NTM(input_dim=3, memory_size=10, memory_dim=5, latent_dim=5, learning_rate=0.001, max_iter=1000)

    # 训练NTM模型
    ntm.fit(X)

    # 使用训练好的NTM进行数据预测
    X_pred = ntm.predict(X)
    print("原始数据：", X[:5])
    print("预测数据：", X_pred[:5])
