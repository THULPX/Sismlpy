import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------- Independent Component Analysis (ICA) 算法 -----------------------------

# 介绍：
# 独立成分分析（Independent Component Analysis, ICA）是一种盲信号分离方法，用于从多个观测信号中提取出相互独立的源信号。ICA 是一种非线性降维技术，常用于信号处理、音频分离（如“盲源分离”问题）以及数据降维。
# ICA 假设观测到的信号是由一些独立的源信号线性混合而成，目的是从这些混合信号中分离出独立的源信号。与 PCA 不同，PCA 是根据方差来选择成分，而 ICA 则关注信号的独立性。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 降维后的数据，形状为 (n_samples, n_components)，其中 n_components 是降维后的维度数。

# 算法步骤：
# 1. 假设数据由若干个独立的成分混合而成。
# 2. 使用最大化非高斯性或最小化高阶统计量（如 kurtosis 或 negentropy）来提取这些独立成分。
# 3. 通过对数据进行旋转，使得数据的成分变得更加独立，从而完成信号分离。

# 主要参数：
# - n_components: 降维后的维度数，默认为 None，表示不限制成分数目。
# - whiten: 是否进行白化处理，默认为 True。白化是去除数据中的冗余信息，使得不同维度的成分具有相同的方差。
# - max_iter: 最大迭代次数，默认为 200。
# - random_state: 随机种子，默认为 None。

class ICAModel:
    def __init__(self, n_components=None, whiten=True, max_iter=200, random_state=None):
        """
        初始化 ICA 模型。

        :param n_components: 降维后的维度数，默认为 None。
        :param whiten: 是否进行白化处理，默认为 True。
        :param max_iter: 最大迭代次数，默认为 200。
        :param random_state: 随机种子，默认为 None。
        """
        self.n_components = n_components
        self.whiten = whiten
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X_train):
        """
        训练 ICA 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 ICA 模型
        self.model = FastICA(n_components=self.n_components, whiten=self.whiten,
                             max_iter=self.max_iter, random_state=self.random_state)
        self.transformed_data = self.model.fit_transform(X_train)
        print("Model trained successfully.")

    def transform(self, X):
        """
        使用训练好的 ICA 模型进行数据降维。

        :param X: 输入数据的特征。
        :return: 降维后的数据。
        """
        transformed_data = self.model.transform(X)
        return transformed_data

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过降维后的数据进行分类评估）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标标签。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        transformed_data = self.transform(X_test)

        # 使用降维后的数据进行简单分类（例如 KNN）评估
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(transformed_data, y_test)

        # 预测
        predictions = knn.predict(transformed_data)

        # 计算准确率
        accuracy = accuracy_score(y_test, predictions)

        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test, predictions)

        # 分类报告
        class_report = classification_report(y_test, predictions)

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

    # 创建 ICA 模型
    ica_model = ICAModel(n_components=2)

    # 训练 ICA 模型
    ica_model.fit(X_train)

    # 评估模型
    ica_model.evaluate(X_test, y_test)
