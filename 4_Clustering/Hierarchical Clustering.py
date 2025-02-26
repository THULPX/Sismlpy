import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ----------------------------- Hierarchical Clustering 算法 -----------------------------

# 介绍：
# 层次聚类（Hierarchical Clustering）是一种无监督学习算法，它通过递归地将数据点进行合并（自底向上）或拆分（自顶向下），逐步构建出一个层次结构（树形图，称为树状图，Dendrogram）。
# 层次聚类有两种主要类型：凝聚型（Agglomerative）和分裂型（Divisive）。凝聚型聚类从每个数据点开始，逐步合并最近的簇，而分裂型聚类从所有数据点的一个簇开始，逐步拆分。
# 层次聚类不需要指定簇的数量，并且可以通过树状图来选择合适的簇数。它适用于分析数据的层次结构关系，但计算复杂度较高，特别是数据集较大时。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 聚类结果：每个数据点所属的簇（标签）。

# 算法步骤：
# 1. 初始化每个数据点为单独一个簇。
# 2. 计算所有簇之间的距离（通常使用欧氏距离或其他距离度量方法）。
# 3. 合并最近的簇，直到达到指定的簇数或距离阈值。
# 4. 通过树状图（Dendrogram）可以选择聚类结果的层次。

# 主要参数：
# - n_clusters: 簇的数量，默认为2。
# - affinity: 距离度量方法，默认为 'euclidean'。
# - linkage: 聚合方式，默认为 'ward'。常见方式包括 'ward', 'complete', 'average', 'single'。
# - memory: 存储路径，默认为 None。

class HierarchicalClusteringModel:
    def __init__(self, n_clusters=3, affinity='euclidean', linkage='ward'):
        """
        初始化层次聚类模型。

        :param n_clusters: 聚类数量，默认为3。
        :param affinity: 距离度量方法，默认为 'euclidean'。
        :param linkage: 聚合方式，默认为 'ward'。
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.model = None

    def fit(self, X_train):
        """
        训练层次聚类模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练层次聚类模型
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity,
                                             linkage=self.linkage)
        self.model.fit(X_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的层次聚类模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（每个样本的簇标签）。
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

        # 由于层次聚类是无监督的，因此需要将预测的簇标签与实际的标签进行对齐。
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

    # 创建层次聚类模型
    hierarchical_model = HierarchicalClusteringModel(n_clusters=3)

    # 训练层次聚类模型
    hierarchical_model.fit(X_train)

    # 评估模型
    hierarchical_model.evaluate(X_test, y_test)
