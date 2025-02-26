import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ----------------------------- DBSCAN 算法 -----------------------------

# 介绍：
# DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，它通过识别具有高密度区域的数据点，形成聚类。与 K-Means 不同，DBSCAN 不需要预定义簇的数量，而是通过两个主要参数：
# - Epsilon（ε）：领域的最大距离，用于定义数据点间的邻域。
# - MinPts：指定邻域内最小的数据点数量，低于该数量的数据点会被视为噪声。
# DBSCAN 的优点是能够识别任意形状的簇，并且对于噪声数据点有较好的鲁棒性。缺点是对于簇的密度分布较为敏感。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 聚类结果：每个数据点所属的簇（标签），噪声点会标记为 -1。

# 算法步骤：
# 1. 对于每个数据点，计算其 ε 邻域内的数据点数量。
# 2. 如果某个数据点的邻域内的数据点数大于或等于 MinPts，则它是核心点，可以成为一个簇的核心。
# 3. 将与核心点邻接的所有数据点划分到同一个簇，继续扩展聚类，直到没有新的点可以加入该簇。
# 4. 如果数据点的邻域内数据点数小于 MinPts，则该点被视为噪声点（标记为 -1）。

# 主要参数：
# - eps: 邻域的最大距离，默认为 0.5。
# - min_samples: 一个簇的最小样本数，默认为 5。
# - metric: 距离度量方法，默认为 'euclidean'。
# - n_jobs: 并行工作的数量，默认为 1。

class DBSCANModel:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', n_jobs=1):
        """
        初始化 DBSCAN 模型。

        :param eps: 邻域的最大距离，默认为 0.5。
        :param min_samples: 每个簇的最小样本数，默认为 5。
        :param metric: 距离度量方法，默认为 'euclidean'。
        :param n_jobs: 并行工作的数量，默认为 1。
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_train):
        """
        训练 DBSCAN 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 DBSCAN 模型
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, n_jobs=self.n_jobs)
        self.model.fit(X_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 DBSCAN 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（每个样本的簇标签，噪声点为 -1）。
        """
        predictions = self.model.fit_predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过调整标签来计算准确度）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量（标签）。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        predictions = self.predict(X_test)

        # 由于 DBSCAN 是无监督的，因此需要将预测的簇标签与实际的标签进行对齐。
        # 使用标签编码器将实际的标签转化为簇标签。
        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(y_test)

        # 计算准确率
        accuracy = accuracy_score(y_test_encoded, predictions)

        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test_encoded, predictions)

        # 分类报告
        class_report = classification_report(y_test_encoded, predictions)

        print(f"Model Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        return accuracy, conf_matrix, class_report


# 示例：使用鸢尾花数据集进行训练和评估
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 DBSCAN 模型
    dbscan_model = DBSCANModel(eps=0.5, min_samples=5)

    # 训练 DBSCAN 模型
    dbscan_model.fit(X_train)

    # 评估模型
    dbscan_model.evaluate(X_test, y_test)
