import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------- Stacking 算法 -----------------------------

# 介绍：
# Stacking（堆叠）是一种集成学习方法，它通过将多个不同的学习器（基学习器）的预测结果作为输入，训练一个元学习器（meta-learner）来进行最终的预测。
# Stacking 与 Bagging 和 Boosting 的区别在于，Stacking 的基学习器的预测结果不是简单地投票或平均，而是将它们的预测作为特征输入到一个新的学习器中（通常是逻辑回归、SVM 或其他分类器），
# 来进行最终的组合预测。Stacking 可以有效地将多个不同类型的模型的优势结合起来，提高预测的准确性。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)。
# 输出：
# - Stacking 模型的预测结果。

# 算法步骤：
# 1. 使用多个基学习器训练不同的模型，并对训练数据进行预测。
# 2. 将基学习器的预测结果作为新的特征，训练一个元学习器（通常是逻辑回归、SVM 等）来进行最终的预测。
# 3. 输出元学习器的预测结果作为模型的最终预测。

# 主要参数：
# - estimators: 基学习器的列表（元组形式），包括每个学习器的名称和实例。
# - final_estimator: 元学习器，默认为 Logistic Regression。
# - cv: 交叉验证次数，默认为 5。
# - n_jobs: 并行工作线程的数量，默认为1。

class StackingModel:
    def __init__(self, estimators=None, final_estimator=None, cv=5, n_jobs=1):
        """
        初始化 Stacking 模型。

        :param estimators: 基学习器列表，默认为决策树和支持向量机。
        :param final_estimator: 元学习器，默认为逻辑回归。
        :param cv: 交叉验证次数，默认为5。
        :param n_jobs: 并行工作线程的数量，默认为1。
        """
        # 默认基学习器：决策树和支持向量机
        if estimators is None:
            estimators = [
                ('decision_tree', DecisionTreeClassifier()),
                ('svm', SVC())
            ]
        self.estimators = estimators
        self.final_estimator = final_estimator if final_estimator is not None else LogisticRegression()
        self.cv = cv
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Stacking 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 初始化并训练 Stacking 模型
        self.model = StackingClassifier(estimators=self.estimators, final_estimator=self.final_estimator,
                                        cv=self.cv, n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Stacking 模型进行预测。

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

    # 创建 Stacking 模型
    stacking_model = StackingModel()

    # 训练 Stacking 模型
    stacking_model.fit(X_train, y_train)

    # 评估模型
    stacking_model.evaluate(X_test, y_test)
