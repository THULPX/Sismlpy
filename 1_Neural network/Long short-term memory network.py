import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# ----------------------------- 长短时记忆网络（LSTM）算法 -----------------------------

# 介绍：
# 长短时记忆网络（LSTM）是一种改进版的循环神经网络（RNN），通过引入“门”机制（遗忘门、输入门、输出门）。
# LSTM解决传统RNN在长序列中梯度消失和梯度爆炸的问题。
# LSTM能够学习长期的依赖关系，广泛应用于语言模型、时间序列预测、语音识别等任务。

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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-4, max_iter=1000):
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
        self.learning_rate = learning_rate  # 降低学习率
        self.max_iter = max_iter

        # Xavier初始化
        def xavier_init(size):
            return np.random.randn(*size) * np.sqrt(2.0 / sum(size))

        # 初始化LSTM模型的权重和偏置
        self.Wf = xavier_init((input_size + hidden_size, hidden_size))  # 遗忘门权重
        self.Wi = xavier_init((input_size + hidden_size, hidden_size))  # 输入门权重
        self.Wc = xavier_init((input_size + hidden_size, hidden_size))  # 细胞状态权重
        self.Wo = xavier_init((input_size + hidden_size, hidden_size))  # 输出门权重
        self.Wy = xavier_init((hidden_size, output_size))  # 输出层权重

        # 初始化偏置为小正数，有助于在训练初期保留更多信息
        self.bf = np.ones((1, hidden_size)) * 0.1  # 遗忘门偏置
        self.bi = np.zeros((1, hidden_size))  # 输入门偏置
        self.bc = np.zeros((1, hidden_size))  # 细胞状态偏置
        self.bo = np.zeros((1, hidden_size))  # 输出门偏置
        self.by = np.zeros((1, output_size))  # 输出层偏置

        # Adam优化器参数（优化器参数调整）
        self.beta1 = 0.8  # 调整beta1
        self.beta2 = 0.999  # 调整beta2
        self.epsilon = 1e-8
        self.m = {k: np.zeros_like(v) for k, v in self.__dict__.items() if k.startswith('W') or k.startswith('b')}
        self.v = {k: np.zeros_like(v) for k, v in self.__dict__.items() if k.startswith('W') or k.startswith('b')}
        self.t = 0

        self.loss_history = []
        self.acc_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _tanh(self, Z):
        return np.tanh(Z)

    def _softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _normalize_data(self, X):
        """数据标准化"""
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    def _forward(self, X):
        batch_size, sequence_length, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        outputs = []
        cell_states = []

        for t in range(sequence_length):
            combined = np.hstack((X[:, t, :], h))

            ft = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
            it = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
            ct = np.tanh(np.dot(combined, self.Wc) + self.bc)
            ot = self._sigmoid(np.dot(combined, self.Wo) + self.bo)

            c = ft * c + it * ct
            h = ot * np.tanh(c)

            outputs.append(h)
            cell_states.append(c)

        # 使用sigmoid确保输出在[0,1]范围内
        y_pred = self._sigmoid(np.dot(h, self.Wy) + self.by)
        return np.array(outputs), np.array(cell_states), y_pred

    def _compute_loss(self, y_pred, y):
        """二元交叉熵损失函数"""
        epsilon = 1e-15  # 防止log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 数值稳定性
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _compute_metrics(self, y_pred, y):
        """动态阈值选择"""
        thresholds = np.arange(0.3, 0.8, 0.1)
        best_f1 = 0
        best_threshold = 0.5

        # 默认使用0.5作为阈值的指标
        pred_classes = (y_pred > best_threshold).astype(int)
        true_classes = (y > 0.5).astype(int)
        best_metrics = (
            accuracy_score(true_classes, pred_classes),
            precision_score(true_classes, pred_classes, zero_division=0),
            recall_score(true_classes, pred_classes, zero_division=0),
            f1_score(true_classes, pred_classes, zero_division=0)
        )

        for threshold in thresholds:
            pred_classes = (y_pred > threshold).astype(int)
            true_classes = (y > 0.5).astype(int)

            current_f1 = f1_score(true_classes, pred_classes, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
                best_metrics = (
                    accuracy_score(true_classes, pred_classes),
                    precision_score(true_classes, pred_classes, zero_division=0),
                    recall_score(true_classes, pred_classes, zero_division=0),
                    current_f1
                )

        return best_metrics

    def _clip_gradients(self, gradient, max_value=3.0):  # 梯度裁剪阈值改为3.0
        """梯度裁剪函数"""
        norm = np.linalg.norm(gradient)
        if norm > max_value:
            return gradient * max_value / norm
        return gradient

    def _adam_update(self, param_name, gradient):
        """Adam优化器更新"""
        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def _backward(self, X, y, outputs, cell_states, y_pred):
        m = y.shape[0]
        sequence_length = len(outputs)

        # 初始化梯度
        gradients = {
            'Wf': 0, 'Wi': 0, 'Wc': 0, 'Wo': 0, 'Wy': 0,
            'bf': 0, 'bi': 0, 'bc': 0, 'bo': 0, 'by': 0
        }

        dWy = self._clip_gradients(np.dot(outputs[-1].T, (y_pred - y)) / m)
        dby = self._clip_gradients(np.sum(y_pred - y, axis=0, keepdims=True) / m)

        gradients['Wy'] = dWy
        gradients['by'] = dby

        dh_next = np.zeros((m, self.hidden_size))
        dc_next = np.zeros((m, self.hidden_size))

        for t in reversed(range(sequence_length)):
            combined = np.hstack((X[:, t, :], outputs[t - 1] if t > 0 else np.zeros_like(outputs[0])))

            ft = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
            it = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
            ct = np.tanh(np.dot(combined, self.Wc) + self.bc)
            ot = self._sigmoid(np.dot(combined, self.Wo) + self.bo)

            if t == sequence_length - 1:
                dh = np.dot(y_pred - y, self.Wy.T) + dh_next
            else:
                dh = dh_next

            dot = dh * np.tanh(cell_states[t]) * ot * (1 - ot)
            dc = dh * ot * (1 - np.tanh(cell_states[t]) ** 2) + dc_next
            dft = dc * cell_states[t - 1] if t > 0 else dc * np.zeros_like(cell_states[0])
            dit = dc * ct
            dct = dc * it

            gradients['Wf'] += self._clip_gradients(np.dot(combined.T, dft) / m)
            gradients['Wi'] += self._clip_gradients(np.dot(combined.T, dit) / m)
            gradients['Wc'] += self._clip_gradients(np.dot(combined.T, dct) / m)
            gradients['Wo'] += self._clip_gradients(np.dot(combined.T, dot) / m)

            gradients['bf'] += self._clip_gradients(np.sum(dft, axis=0, keepdims=True) / m)
            gradients['bi'] += self._clip_gradients(np.sum(dit, axis=0, keepdims=True) / m)
            gradients['bc'] += self._clip_gradients(np.sum(dct, axis=0, keepdims=True) / m)
            gradients['bo'] += self._clip_gradients(np.sum(dot, axis=0, keepdims=True) / m)

            dh_next = np.dot(dft, self.Wf.T)[:, self.input_size:] + \
                      np.dot(dit, self.Wi.T)[:, self.input_size:] + \
                      np.dot(dct, self.Wc.T)[:, self.input_size:] + \
                      np.dot(dot, self.Wo.T)[:, self.input_size:]
            dc_next = dc * ft

        # 使用Adam优化器更新参数
        for param_name, gradient in gradients.items():
            update = self._adam_update(param_name, gradient)
            setattr(self, param_name, getattr(self, param_name) - update)

        loss = self._compute_loss(y_pred, y)
        accuracy, precision, recall, f1 = self._compute_metrics(y_pred, y)

        self.loss_history.append(loss)
        self.acc_history.append(accuracy)
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

    def fit(self, X, y):
        """训练模型"""
        # 数据预处理
        X = self._normalize_data(X)

        for _ in range(self.max_iter):
            outputs, cell_states, y_pred = self._forward(X)
            self._backward(X, y, outputs, cell_states, y_pred)

    def predict(self, X):
        _, _, y_pred = self._forward(X)
        return y_pred


# 示例：创建LSTM模型并训练
if __name__ == "__main__":
    # 生成二元分类的示例数据
    np.random.seed(42)  # 设置随机种子以确保可重复性

    # 生成序列数据
    X = np.random.randn(100, 10, 5)  # 100个样本，序列长度10，每个时间步5维输入

    # 生成二元分类标签 (0或1)
    y = np.random.randint(0, 2, (100, 1))

    # 创建和训练模型
    lstm = LSTM(input_size=5, hidden_size=32, output_size=1, learning_rate=1e-4, max_iter=1200)
    lstm.fit(X, y)

    # 可视化训练效果
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(lstm.loss_history, label="Loss", color='blue')
    plt.title("Binary Cross-Entropy Loss during Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 绘制评估指标曲线
    plt.subplot(1, 2, 2)
    plt.plot(lstm.acc_history, label="Accuracy", color='blue')
    plt.plot(lstm.precision_history, label="Precision", color='green')
    plt.plot(lstm.recall_history, label="Recall", color='red')
    plt.plot(lstm.f1_history, label="F1 Score", color='purple')
    plt.title("Evaluation Metrics during Training")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
