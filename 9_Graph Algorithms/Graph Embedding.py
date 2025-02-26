import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# ----------------------------- Graph Embedding -----------------------------

# 介绍：
# 图嵌入（Graph Embedding）是一种将图中的节点、边或整个图结构转换为低维向量表示的技术。目标是将图的结构信息保留在低维空间中，使得通过这些向量表示能够进行图的分析与预测任务。常见的图嵌入方法包括DeepWalk、node2vec等。图嵌入可以用于节点分类、图分类、链路预测等任务。

# 输入输出：
# 输入：
# - G: 图结构，可以是邻接矩阵、边列表等形式。
# - embedding_dim: 嵌入的维度。
# 输出：
# - node_embeddings: 图中每个节点的低维嵌入表示。

# 算法步骤：
# 1. 通过图的邻接矩阵或邻接列表表示图。
# 2. 使用图嵌入方法（如DeepWalk、node2vec等）学习节点的低维表示。
# 3. 返回每个节点的嵌入向量。

# 主要参数：
# - G: 图结构。
# - embedding_dim: 嵌入维度。

class GraphEmbedding:
    def __init__(self, graph, embedding_dim=64, walk_length=10, num_walks=80, window_size=5, workers=4):
        """
        初始化图嵌入。

        :param graph: 图结构（邻接矩阵）。
        :param embedding_dim: 嵌入空间的维度。
        :param walk_length: 每次随机游走的长度。
        :param num_walks: 每个节点生成的随机游走次数。
        :param window_size: Skip-gram模型的窗口大小。
        :param workers: 使用的并行线程数。
        """
        self.graph = graph  # 邻接矩阵
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.walk_length = walk_length  # 每次随机游走的长度
        self.num_walks = num_walks  # 每个节点的随机游走次数
        self.window_size = window_size  # Skip-gram的窗口大小
        self.workers = workers  # 并行数

    def generate_walks(self):
        """
        生成随机游走序列。
        :return: 随机游走序列。
        """
        walks = []
        nodes = list(self.graph.keys())

        for node in nodes:
            for _ in range(self.num_walks):
                walk = self.random_walk(node)
                walks.append(walk)

        return walks

    def random_walk(self, start_node):
        """
        从一个节点出发，进行随机游走。
        :param start_node: 起始节点。
        :return: 随机游走路径。
        """
        walk = [start_node]
        current_node = start_node

        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph[current_node])
            if len(neighbors) > 0:
                current_node = np.random.choice(neighbors)
                walk.append(current_node)
            else:
                break
        return walk

    def learn_embeddings(self, walks):
        """
        学习节点的低维嵌入。
        :param walks: 随机游走路径。
        :return: 学到的节点嵌入。
        """
        # 使用Skip-gram模型进行学习（类似Word2Vec）
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.graph), self.embedding_dim, input_length=self.walk_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(self.embedding_dim)
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # 创建训练数据
        X_train, y_train = self.create_training_data(walks)

        # 训练嵌入模型
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # 返回节点嵌入
        embeddings = model.layers[0].get_weights()[0]
        return embeddings

    def create_training_data(self, walks):
        """
        将随机游走转换为训练数据。
        :param walks: 随机游走序列。
        :return: 训练数据。
        """
        # 创建skip-gram模型的训练数据
        X_train = []
        y_train = []

        for walk in walks:
            for i in range(1, len(walk)):
                context = walk[i - 1]
                target = walk[i]
                X_train.append(context)
                y_train.append(target)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train

    def fit(self):
        """
        执行图嵌入并返回节点的嵌入表示。
        :return: 节点嵌入。
        """
        walks = self.generate_walks()
        embeddings = self.learn_embeddings(walks)
        return embeddings


# ----------------------------- 示例：使用Graph Embedding进行节点表示学习 -----------------------------

# 创建一个简单的图结构（邻接表）
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2]
}

# 初始化GraphEmbedding模型
graph_embedding = GraphEmbedding(graph=graph, embedding_dim=16, walk_length=10, num_walks=20)

# 执行图嵌入
node_embeddings = graph_embedding.fit()

# 输出节点嵌入表示
print("节点嵌入表示：")
for node, embedding in enumerate(node_embeddings):
    print(f"节点 {node}: {embedding}")
