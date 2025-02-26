import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------- Bagging 算法 -----------------------------

# 介绍：
# Bagging（Bootstrap Aggregating）是一种集成学习方法，旨在通过训练多个基学习器并将其预测结果进行平均或投票来提升模型的准确性和稳定性。Bagging的核心思想是：
# 1. 从训练数据集中随机有放回地采样，生成多个不同的训练子集。
# 2. 对每个子集训练一个独立的基学习器。
# 3. 将所有基学习器的预测结果进行组合，通常使用平均值（回归任务）或多数投票（分类任务）来得出最终的预测结果。
# Bagging常用于减少高方差模型（例如决策树）的过拟合。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)，类别值。
# 输出：
# - Bagging模型的预测结果。

# 算法步骤：
# 1. 从训练数据中随机有放回地采样，生成多个训练子集。
# 2. 对每个子集训练一个基学习器（例如决策树）。
# 3. 对所有训练好的基学习器进行预测，并将结果进行组合（回归任务使用均值，分类任务使用多数投票）。
# 4. 输出最终的预测结果。

# 主要参数：
# - base_estimator: 基学习器，默认为决策树。
# - n_estimators: 基学习器的数量，默认为10。
# - max_samples: 每个基学习器使用的样本数，默认为1.0（表示使用全部样本）。
# - max_features: 每个基学习器使用的特征数，默认为1.0（表示使用全部特征）。
# - n_jobs: 并行工作的作业数量，默认为1。

class BaggingModel:
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, n_jobs=1):
        """
        初始化 Bagging 模型。

        :param base_estimator: 基学习器，默认为决策树。
        :param n_estimators: 基学习器的数量，默认为10。
        :param max_samples: 每个基学习器使用的样本数，默认为1.0（表示使用全部样本）。
        :param max_features: 每个基学习器使用的特征数，默认为1.0（表示使用全部特征）。
        :param n_jobs: 并行工作的作业数量，默认为1。
        """
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Bagging 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 初始化并训练 Bagging 模型
        self.model = BaggingClassifier(base_estimator=self.base_estimator, n_estimators=self.n_estimators,
                                      max_samples=self.max_samples, max_features=self.max_features,
                                      n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Bagging 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（分类标签）。
        """
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量（标签）。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        predictions = self.predict(X_test)

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

    # 创建 Bagging 模型
    bagging_model = BaggingModel(base_estimator=DecisionTreeClassifier(), n_estimators=50, n_jobs=-1)

    # 训练 Bagging 模型
    bagging_model.fit(X_train, y_train)

    # 评估模型
    bagging_model.evaluate(X_test, y_test)
