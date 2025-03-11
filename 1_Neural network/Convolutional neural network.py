import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ----------------------------- 卷积神经网络（CNN）算法 -----------------------------

# 介绍：
# 卷积神经网络（CNN）是一种用于处理具有网格状拓扑的数据的深度神经网络（如图像）。
# CNN通过层叠卷积层、池化层和全连接层来自动提取特征。
# 卷积层能够有效捕捉局部特征，而池化层则用于下采样和特征缩放。
# CNN通常用于图像分类、物体检测等任务。

# 输入输出：
# 输入：
# - X: 输入数据，通常是图像数据，形状为 (n_samples, height, width, channels)。
# - y: 目标标签，形状为 (n_samples, n_classes)。
# 输出：
# - 模型训练好的权重和偏置，以及预测结果。

# 算法步骤：
# 1. 初始化卷积层、池化层和全连接层的权重和偏置。
# 2. 前向传播：依次计算卷积层、池化层和全连接层的输出。
# 3. 计算损失函数（交叉熵损失或均方误差）。
# 4. 反向传播：计算损失对每一层的梯度，更新权重和偏置。
# 5. 重复步骤2到步骤4，直到损失收敛。

# 主要参数：
# - learning_rate：学习率，用于权重更新的步长。
# - max_iter：最大迭代次数。
# - filter_size：卷积核的大小。
# - num_filters：每一层的卷积核个数。
# - pool_size：池化层的大小。

class CNN:
    def __init__(self, filter_size=2, num_filters=[8], pool_size=2, learning_rate=0.01, max_iter=200):
        """
        初始化卷积神经网络CNN模型。

        :param filter_size: 卷积核的大小
        :param num_filters: 每一层卷积层的滤波器数量
        :param pool_size: 池化层的大小
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.biases = []
        
        # 用于记录训练历史
        self.history = {
            'loss': [],
            'accuracy': []
        }

    def _initialize_weights(self, input_shape):
        """初始化权重"""
        _, height, width, channels = input_shape
        
        # 计算每层的输出大小
        curr_h, curr_w = height, width
        
        for i, num_filter in enumerate(self.num_filters):
            # 计算卷积后的尺寸
            curr_h = curr_h - self.filter_size + 1
            curr_w = curr_w - self.filter_size + 1
            
            # 计算池化后的尺寸
            curr_h = curr_h // self.pool_size
            curr_w = curr_w // self.pool_size
            
            print(f"Layer {i} output shape: ({curr_h}, {curr_w})")
            
            # He初始化
            if i == 0:
                fan_in = self.filter_size * self.filter_size * channels
            else:
                fan_in = self.filter_size * self.filter_size * self.num_filters[i-1]
            
            scale = np.sqrt(2. / fan_in)
            W = np.random.randn(self.filter_size, self.filter_size, 
                              channels if i == 0 else self.num_filters[i-1], 
                              num_filter) * scale
            b = np.zeros((1, 1, 1, num_filter))
            
            self.weights.append(W)
            self.biases.append(b)
            channels = num_filter  # 更新下一层的输入通道数

        # 计算全连接层的输入维度
        fc_input_size = curr_h * curr_w * self.num_filters[-1]
        print(f"FC input size: {fc_input_size}")
        
        # He初始化全连接层
        fc_scale = np.sqrt(2. / fc_input_size)
        self.fc_weights = np.random.randn(fc_input_size, 10) * fc_scale
        self.fc_biases = np.zeros((1, 10))

    def _conv2d(self, X, W, b):
        """卷积运算的向量化实现"""
        n, h, w, c = X.shape
        kh, kw, _, out_c = W.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        
        # 计算卷积
        Z = np.zeros((n, out_h, out_w, out_c))
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, i:i+kh, j:j+kw, :]
                for k in range(out_c):
                    Z[:, i, j, k] = np.sum(patch * W[:, :, :, k], axis=(1, 2, 3)) + b[0, 0, 0, k]
        return Z

    def _pool(self, X):
        """最大池化的向量化实现"""
        n, h, w, c = X.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size
        
        # 直接进行最大池化
        Z = np.zeros((n, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size
                Z[:, i, j, :] = np.max(X[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        return Z

    def _forward(self, X):
        """前向传播"""
        self.cache = {'X': X}
        out = X
        
        # 卷积和池化层
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # 卷积
            conv = self._conv2d(out, W, b)
            self.cache[f'conv{i}'] = conv
            
            # ReLU
            relu = np.maximum(0, conv)
            self.cache[f'relu{i}'] = relu
            
            # 池化
            out = self._pool(relu)
            self.cache[f'pool{i}'] = out
        
        # 全连接层
        out_flat = out.reshape(out.shape[0], -1)
        self.cache['flatten'] = out_flat
        fc_out = np.dot(out_flat, self.fc_weights) + self.fc_biases
        
        return fc_out

    def _backward(self, dout):
        """反向传播"""
        n = dout.shape[0]
        
        # 全连接层的梯度
        dfc = np.dot(self.cache['flatten'].T, dout) / n
        db_fc = np.sum(dout, axis=0, keepdims=True) / n
        
        # 更新全连接层
        self.fc_weights -= self.learning_rate * dfc
        self.fc_biases -= self.learning_rate * db_fc
        
        # 传递给展平层的梯度
        dout = np.dot(dout, self.fc_weights.T)
        dout = dout.reshape(self.cache[f'pool{len(self.weights)-1}'].shape)
        
        # 反向传播通过卷积层
        for i in range(len(self.weights)-1, -1, -1):
            # 池化层梯度
            dpool = np.zeros_like(self.cache[f'relu{i}'])
            for b in range(n):
                for h in range(dout.shape[1]):
                    for w in range(dout.shape[2]):
                        h_start = h * self.pool_size
                        h_end = h_start + self.pool_size
                        w_start = w * self.pool_size
                        w_end = w_start + self.pool_size
                        patch = self.cache[f'relu{i}'][b, h_start:h_end, w_start:w_end, :]
                        mask = patch == patch.max(axis=(0, 1), keepdims=True)
                        dpool[b, h_start:h_end, w_start:w_end, :] = mask * dout[b, h, w, :].reshape(1, 1, -1)
            
            # ReLU梯度
            drelu = dpool * (self.cache[f'conv{i}'] > 0)
            
            # 卷积层梯度
            if i > 0:
                prev_output = self.cache[f'pool{i-1}']
            else:
                prev_output = self.cache['X']
            
            dW = np.zeros_like(self.weights[i])
            db = np.zeros_like(self.biases[i])
            
            for b in range(n):
                for h in range(drelu.shape[1] - self.filter_size + 1):
                    for w in range(drelu.shape[2] - self.filter_size + 1):
                        patch = prev_output[b, h:h+self.filter_size, w:w+self.filter_size, :]
                        for c_out in range(drelu.shape[3]):
                            dW[:, :, :, c_out] += patch * drelu[b, h, w, c_out]
                            db[0, 0, 0, c_out] += drelu[b, h, w, c_out]
            
            # 更新卷积层权重
            self.weights[i] -= self.learning_rate * dW / n
            self.biases[i] -= self.learning_rate * db / n
            
            dout = drelu

    def evaluate(self, X, y):
        """评估模型性能"""
        predictions = self.predict(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return accuracy

    def fit(self, X, y, batch_size=32, validation_data=None):
        """训练模型"""
        print("Input shape:", X.shape)
        self._initialize_weights(X.shape)
        n_samples = X.shape[0]
        
        # 记录每个epoch的损失和准确率
        epoch_losses = []
        epoch_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.max_iter):
            # Mini-batch训练
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # 前向传播
                out = self._forward(batch_X)
                
                # 计算softmax和交叉熵损失
                exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
                softmax_out = exp_out / np.sum(exp_out, axis=1, keepdims=True)
                loss = -np.sum(batch_y * np.log(softmax_out + 1e-8)) / batch_size
                
                # 反向传播
                dout = softmax_out - batch_y
                self._backward(dout)
                
                total_loss += loss
                n_batches += 1
            
            # 计算当前epoch的平均损失和准确率
            avg_loss = total_loss / n_batches
            accuracy = self.evaluate(X, y)
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            
            # 如果提供了验证集，计算验证集上的性能
            if validation_data is not None:
                val_X, val_y = validation_data
                val_pred = self.predict(val_X)
                val_loss = -np.sum(val_y * np.log(val_pred + 1e-8)) / val_X.shape[0]
                val_acc = self.evaluate(val_X, val_y)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                if validation_data is not None:
                    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # 保存训练历史
        self.history['loss'] = epoch_losses
        self.history['accuracy'] = epoch_accuracies
        if validation_data is not None:
            self.history['val_loss'] = val_losses
            self.history['val_accuracy'] = val_accuracies
        
        return self.history

    def plot_training_history(self):
        """绘制训练历史曲线"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        # 创建一个2x1的子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        ax1.plot(epochs, self.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in self.history:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(epochs, self.history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in self.history:
            ax2.plot(epochs, self.history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X, y_true):
        """绘制混淆矩阵"""
        # 获取预测结果
        y_pred = self.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # 绘制混淆矩阵热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))

    def predict(self, X):
        """预测"""
        out = self._forward(X)
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=1, keepdims=True)


# 示例：创建CNN模型并训练
if __name__ == "__main__":
    # 加载数据
    X, y = load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8, 1) / 16.0  # 归一化到[0,1]范围
    y = np.eye(10)[y]  # One-hot编码
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建和训练模型
    cnn = CNN(filter_size=2, num_filters=[8], pool_size=2, learning_rate=0.01, max_iter=200)
    history = cnn.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    # 绘制训练历史
    cnn.plot_training_history()
    
    # 绘制混淆矩阵
    cnn.plot_confusion_matrix(X_test, y_test)
    
    # 评估模型
    test_accuracy = cnn.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")