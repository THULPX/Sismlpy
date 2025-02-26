import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ----------------------------- Gaussian Mixture Models (GMM) 算法 -----------------------------

# 介绍：
# 高斯混合模型（Gaussian Mixture Model, GMM）是一种概率模型，假设所有的数据点都是由若干个高斯分布（正态分布）生成的。GMM 是一种生成式模型，它通过最大化似然估计，估计数据点属于每个高斯分布的概率。
# GMM 可以用于聚类分析，并且与 K-Means 不同，它能够处理具有不同协方差的簇，适用于数据中存在多种不同分布的情况。GMM 常用于需要捕捉簇内不同变异的任务中。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 聚类结果：每个数据点所属的簇（标签），以及各个数据点属于每个簇的概率。

# 算法步骤：
# 1. 假设数据点来自于 K 个高斯分布，每个高斯分布由均值和协方差描述。
# 2. 通过期望最大化（EM）算法，估计高斯分布的参数，包括均值、协方差和混合系数。
# 3. 计算每个数据点属于每个高斯分布的概率。
# 4. 通过迭代的方式优化高斯分布的参数，直到收敛。

# 主要参数：
# - n_components: 高斯分布的数量，默认为 1。
# - covariance_type: 协方差类型，默认为 'full'。可选 'full', 'tied', 'diag', 'spherical'。
# - max_iter: 最大迭代次数，默认为 100。
# - random_state: 随机种子，默认为 None。

class GMMModel:
    def __init__(self, n_components=3, covariance_type='full', max_iter=100, random_state=None):
        """
        初始化 GMM 模型。

        :param n_components: 高斯分布的数量，默认为3。
        :param covariance_type: 协方差类型，默认为 'full'。
        :param max_iter: 最大迭代次数，默认为100。
        :param random_state: 随机种子，默认为 None。
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X_train):
        """
        训练 GMM 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 GMM 模型
        self.model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                     max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 GMM 模型进行预测。

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

        # 由于 GMM 是无监督的，因此需要将预测的簇标签与实际的标签进行对齐。
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

    # 创建 GMM 模型
    gmm_model = GMMModel(n_components=3)

    # 训练 GMM 模型
    gmm_model.fit(X_train)

    # 评估模型
    gmm_model.evaluate(X_test, y_test)
