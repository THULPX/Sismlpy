import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------- Logistic Regression 算法 -----------------------------

# 介绍：
# 逻辑回归（Logistic Regression）是一种广泛使用的分类模型，通常用于二分类问题。它通过使用sigmoid函数（逻辑函数）将线性回归模型的输出映射到[0,1]范围内，
# 从而输出事件发生的概率。尽管名字中包含“回归”二字，逻辑回归实际上是一个分类算法，而非回归算法。
# 逻辑回归的目标是通过最大化似然函数来估计模型参数，使得预测概率与实际标签之间的误差最小。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)，类别值为0或1。
# 输出：
# - Logistic回归模型的预测结果。

# 算法步骤：
# 1. 将输入特征与目标变量之间的关系建模为：P(y=1|X) = sigmoid(Xw + b)，其中sigmoid(z) = 1 / (1 + exp(-z))。
# 2. 通过最大化对数似然函数来估计模型参数w和b，常用优化方法有梯度下降法。
# 3. 模型训练完成后，通过阈值判断将预测概率转化为分类结果（通常阈值为0.5，即当P(y=1|X) >= 0.5时预测为1，反之为0）。
# 4. 对新样本进行预测，得到分类标签。

# 主要参数：
# - C: 正则化强度的倒数，越小表示正则化越强，默认为1.0。
# - penalty: 正则化类型，选择'L2'（默认）或'L1'。
# - solver: 优化算法的选择，常用的有'liblinear', 'newton-cg', 'lbfgs', 'saga'等。
# - max_iter: 最大迭代次数，默认为100。

class LogisticRegressionModel:
    def __init__(self, C=1.0, penalty='l2', solver='liblinear', max_iter=100):
        """
        初始化 Logistic 回归模型。

        :param C: 正则化强度的倒数，越小正则化越强，默认为1.0。
        :param penalty: 正则化类型，默认为'l2'，可选'l1'。
        :param solver: 优化算法，默认为'liblinear'。
        :param max_iter: 最大迭代次数，默认为100。
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Logistic 回归模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量（标签）。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 Logistic 回归模型
        self.model = LogisticRegression(C=self.C, penalty=self.penalty, solver=self.solver, max_iter=self.max_iter)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Logistic 回归模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（分类标签）。
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


# 示例：使用鸢尾花数据集进行训练和评估（虽然是多分类问题，但可以选择只分类某一类别来演示二分类）
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # 这里只选择类别0作为二分类任务（Setosa vs. Not Setosa）

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 Logistic 回归模型
    logistic_model = LogisticRegressionModel(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

    # 训练 Logistic 回归模型
    logistic_model.fit(X_train, y_train)

    # 评估模型
    logistic_model.evaluate(X_test, y_test)
