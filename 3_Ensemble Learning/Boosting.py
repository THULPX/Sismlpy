import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------- Boosting 算法 -----------------------------

# 介绍：
# Boosting 是一种集成学习方法，旨在通过训练多个弱学习器（例如简单的决策树）并将它们组合起来，逐步提高模型的预测性能。与 Bagging 不同，Boosting 通过调整训练样本的权重来关注前一轮模型预测错误的样本。
# 通过这种方式，Boosting 可以将多个弱学习器组合成一个强学习器。常见的 Boosting 算法包括 AdaBoost、Gradient Boosting 和 XGBoost。
# Boosting 能有效减少偏差，并通过强力集成学习器提高模型的性能。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)。
# 输出：
# - Boosting 模型的预测结果。

# 算法步骤：
# 1. 初始化训练数据的样本权重为相等的值。
# 2. 在每一轮训练中，训练一个弱学习器（例如决策树），并通过当前样本的加权损失函数来优化该弱学习器。
# 3. 更新样本的权重，对于分类错误的样本增加权重，使得下次迭代时更多地关注这些样本。
# 4. 将所有弱学习器的预测结果进行加权组合，得到最终的预测结果。

# 主要参数：
# - base_estimator: 基学习器，默认为决策树。
# - n_estimators: 基学习器的数量，默认为50。
# - learning_rate: 学习率，控制每一轮模型的贡献大小，默认为1.0。
# - algorithm: Boosting算法的类型，'SAMME' 或 'SAMME.R'，默认为 'SAMME.R'。
# - random_state: 随机种子，默认为 None。

class BoostingModel:
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
        """
        初始化 Boosting 模型。

        :param base_estimator: 基学习器，默认为决策树。
        :param n_estimators: 基学习器的数量，默认为50。
        :param learning_rate: 学习率，默认为1.0。
        :param algorithm: Boosting 算法类型，默认为 'SAMME.R'。
        :param random_state: 随机种子，默认为 None。
        """
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Boosting 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 初始化并训练 Boosting 模型
        self.model = AdaBoostClassifier(base_estimator=self.base_estimator, n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate, algorithm=self.algorithm,
                                        random_state=self.random_state)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Boosting 模型进行预测。

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

    # 选择前两个类别（例如Setosa和Versicolor）进行二分类任务
    y = (y != 2).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 Boosting 模型
    boosting_model = BoostingModel(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0)

    # 训练 Boosting 模型
    boosting_model.fit(X_train, y_train)

    # 评估模型
    boosting_model.evaluate(X_test, y_test)
