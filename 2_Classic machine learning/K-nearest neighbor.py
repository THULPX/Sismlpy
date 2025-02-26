import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------- K-Nearest Neighbors (KNN) 算法 -----------------------------

# 介绍：
# K-Nearest Neighbors（KNN）是一种监督学习算法，用于分类和回归任务。KNN的基本思想是根据样本在特征空间中的相似度来进行预测。
# 对于分类任务，KNN通过找出距离待预测样本最近的K个样本，并根据这些邻居的标签来进行投票，最终确定待预测样本的标签。
# 对于回归任务，KNN则是通过计算K个最近邻的平均值来进行预测。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - KNN 模型的预测结果。

# 算法步骤：
# 1. 选择一个整数K，作为最近邻的数量。
# 2. 对于每一个待预测的样本，计算它与训练集中所有样本之间的距离。
# 3. 选择距离最近的K个邻居。
# 4. 对于分类任务，根据K个邻居的类别标签进行投票，返回出现次数最多的标签作为预测结果。
#    对于回归任务，返回K个邻居标签的平均值作为预测结果。

# 主要参数：
# - n_neighbors: K值，即选择的邻居数。
# - metric: 距离度量方式，常见的是欧氏距离（'euclidean'）。
# - algorithm: 计算最近邻的算法，支持'auto', 'ball_tree', 'kd_tree', 'brute'。

class KNNModel:
    def __init__(self, n_neighbors=5, metric='minkowski'):
        """
        初始化 K-Nearest Neighbors 模型。

        :param n_neighbors: K值，即选择的邻居数。
        :param metric: 距离度量方式，默认为'minkowski'，也可以选择其他方式如'euclidean'。
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 KNN 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 KNN 模型
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 KNN 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        # 使用训练好的模型进行预测
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的标签。
        :return: 模型准确率。
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy


# 示例：使用鸢尾花数据集进行训练和评估
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 KNN 模型
    knn_model = KNNModel(n_neighbors=5, metric='minkowski')

    # 训练 KNN 模型
    knn_model.fit(X_train, y_train)

    # 评估模型
    knn_model.evaluate(X_test, y_test)
