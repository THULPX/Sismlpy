import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ----------------------------- K-Means 算法 -----------------------------

# 介绍：
# K-Means 是一种经典的无监督学习算法，旨在将数据集划分为 K 个簇（clusters），每个簇由中心点（centroid）表示。K-Means 算法通过迭代更新簇中心，并分配每个数据点到最近的簇中心，直到簇中心不再发生变化或达到最大迭代次数。
# K-Means 算法通常用于数据聚类分析，且适用于大规模数据集。其优点是计算效率较高，但缺点是需要事先指定簇的数量 K，并且对初始值和离群点比较敏感。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - K: 簇的数量，预定义的超参数。
# 输出：
# - 聚类结果：每个数据点所属的簇（标签）。

# 算法步骤：
# 1. 随机选择 K 个数据点作为初始簇中心。
# 2. 将每个数据点分配到最近的簇中心。
# 3. 更新簇中心为当前簇内所有数据点的均值。
# 4. 重复步骤 2 和 3，直到簇中心不再变化或达到最大迭代次数。

# 主要参数：
# - n_clusters: 簇的数量，默认为 3。
# - max_iter: 最大迭代次数，默认为 300。
# - n_init: 初始中心选择的次数，默认为 10。
# - random_state: 随机种子，默认为 None。

class KMeansModel:
    def __init__(self, n_clusters=3, max_iter=300, n_init=10, random_state=None):
        """
        初始化 K-Means 模型。

        :param n_clusters: 簇的数量，默认为3。
        :param max_iter: 最大迭代次数，默认为300。
        :param n_init: 初始中心选择的次数，默认为10。
        :param random_state: 随机种子，默认为 None。
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.model = None

    def fit(self, X_train):
        """
        训练 K-Means 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 K-Means 模型
        self.model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, n_init=self.n_init,
                            random_state=self.random_state)
        self.model.fit(X_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 K-Means 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（每个样本的簇标签）。
        """
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过调整标签来计算准确度）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量（标签）。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        predictions = self.predict(X_test)

        # 由于 K-Means 是无监督的，因此需要将预测的簇标签与实际的标签进行对齐。
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

    # 创建 K-Means 模型
    kmeans_model = KMeansModel(n_clusters=3)

    # 训练 K-Means 模型
    kmeans_model.fit(X_train)

    # 评估模型
    kmeans_model.evaluate(X_test, y_test)
