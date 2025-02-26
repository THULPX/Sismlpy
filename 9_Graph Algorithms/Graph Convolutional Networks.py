import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import networkx as nx
import scipy.sparse as sp

# ----------------------------- Graph Convolutional Networks (GCN) -----------------------------

# 介绍：
# 图卷积网络（GCN）是一类用于图数据的深度学习模型，通过卷积操作来学习节点的表示。GCN旨在通过节点之间的邻接关系，将图结构的信息传递到节点特征中，从而实现图上节点的表示学习。GCN的核心思想是聚合每个节点的邻居节点的信息，结合节点的自身特征，来更新节点的表示。GCN常用于图分类、节点分类、链接预测等任务。

# 输入输出：
# 输入：
# - A: 邻接矩阵。
# - X: 节点特征矩阵。
# - n_classes: 输出类别数（对于分类任务）。
# 输出：
# - 节点的表示：通过GCN模型得到的节点嵌入表示。

# 算法步骤：
# 1. 初始化节点特征X。
# 2. 使用图卷积层进行特征聚合，利用邻接矩阵A计算每个节点的新的表示。
# 3. 通过多层图卷积层进一步处理信息。
# 4. 最后，根据节点表示进行分类或其他任务。

# 主要参数：
# - A: 邻接矩阵（图的结构）。
# - X: 节点特征矩阵。
# - n_classes: 类别数，用于分类任务。

class GraphConvolutionLayer(layers.Layer):
    def __init__(self, output_dim, activation=None):
        """
        初始化图卷积层。

        :param output_dim: 输出特征维度。
        :param activation: 激活函数。
        """
        super(GraphConvolutionLayer, self).__init__()
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        # 初始化权重矩阵W
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[0][-1], self.output_dim),
            initializer='glorot_uniform',
        )
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs):
        A, X = inputs
        # 图卷积操作：A * X * W
        output = tf.matmul(A, X)  # A * X
        output = tf.matmul(output, self.W)  # A * X * W

        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN(tf.keras.Model):
    def __init__(self, n_classes, hidden_dim=32):
        """
        初始化GCN模型。

        :param n_classes: 输出类别数。
        :param hidden_dim: 隐藏层维度。
        """
        super(GCN, self).__init__()
        self.hidden_layer = GraphConvolutionLayer(output_dim=hidden_dim, activation=tf.nn.relu)
        self.output_layer = GraphConvolutionLayer(output_dim=n_classes, activation=None)

    def call(self, inputs):
        A, X = inputs
        # 第一层图卷积
        X = self.hidden_layer([A, X])
        # 第二层图卷积（输出层）
        X = self.output_layer([A, X])
        return X


# ----------------------------- 示例：使用GCN进行节点分类 -----------------------------

# 创建一个简单的图结构
G = nx.erdos_renyi_graph(10, 0.5)  # 生成一个10节点的随机图
A = nx.adjacency_matrix(G).todense()  # 邻接矩阵
A = np.array(A)
A = A + np.eye(A.shape[0])  # 自连接

# 创建节点特征矩阵，随机生成
X = np.random.rand(10, 5)  # 假设每个节点有5维特征

# 节点分类标签
Y = np.random.randint(0, 2, 10)  # 假设有2类标签（0和1）

# 将邻接矩阵归一化
D = np.diag(np.sum(A, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt  # 归一化的邻接矩阵

# 转换为TensorFlow格式
A_hat = tf.convert_to_tensor(A_hat, dtype=tf.float32)
X = tf.convert_to_tensor(X, dtype=tf.float32)

# 创建GCN模型
model = GCN(n_classes=2, hidden_dim=16)

# 编译模型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# 训练模型
model.fit([A_hat, X], Y, epochs=100, batch_size=1)

# 获取节点的预测
predictions = model([A_hat, X])
print("节点预测类别：", tf.argmax(predictions, axis=1))
