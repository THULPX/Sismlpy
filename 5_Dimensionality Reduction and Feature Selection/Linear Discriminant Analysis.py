import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ----------------------------- Linear Discriminant Analysis (LDA) 算法 -----------------------------

# 介绍：
# 线性判别分析（Linear Discriminant Analysis, LDA）是一种监督学习算法，旨在通过找到数据中各类别之间的最大可分性，来减少特征的维度。与 PCA（主成分分析）不同，LDA 不仅考虑数据的方差，还考虑了类别标签信息。
# LDA 试图通过最大化类间散度与类内散度的比率，找到最能区分不同类别的投影方向。它常用于模式识别、分类任务和降维。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标标签，形状为 (n_samples,)。
# 输出：
# - 降维后的数据，形状为 (n_samples, n_components)，其中 n_components 是降维后的维度数。
# - 类别预测结果。

# 算法步骤：
# 1. 计算类内散度矩阵和类间散度矩阵。
# 2. 计算类内散度矩阵的逆矩阵与类间散度矩阵的乘积，找到最大可分方向。
# 3. 将数据投影到找到的方向上，实现降维。
# 4. 在降维后的空间中使用分类器（例如线性分类器）进行预测。

# 主要参数：
# - n_components: 降维后的维度数，默认为 1。
# - solver: 求解器，默认为 'svd'。可以选择 'lsqr' 或 'eigen'。
# - priors: 类别的先验概率，默认为 None。
# - shrinkage: 收缩参数，默认为 None（只有 solver='lsqr' 时使用）。

class LDAModel:
    def __init__(self, n_components=1, solver='svd', priors=None, shrinkage=None):
        """
        初始化 LDA 模型。

        :param n_components: 降维后的维度数，默认为 1。
        :param solver: 求解器，默认为 'svd'。
        :param priors: 类别的先验概率，默认为 None。
        :param shrinkage: 收缩参数，默认为 None（只有 solver='lsqr' 时使用）。
        """
        self.n_components = n_components
        self.solver = solver
        self.priors = priors
        self.shrinkage = shrinkage
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 LDA 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标标签。
        """
        # 初始化并训练 LDA 模型
        self.model = LinearDiscriminantAnalysis(n_components=self.n_components, solver=self.solver,
                                                priors=self.priors, shrinkage=self.shrinkage)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def transform(self, X):
        """
        使用训练好的 LDA 模型进行数据降维。

        :param X: 输入数据的特征。
        :return: 降维后的数据。
        """
        transformed_data = self.model.transform(X)
        return transformed_data

    def predict(self, X):
        """
        使用训练好的 LDA 模型进行预测。

        :param X: 输入数据的特征。
        :return: 类别预测结果。
        """
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过分类评估指标）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标标签。
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

    # 创建 LDA 模型
    lda_model = LDAModel(n_components=2)

    # 训练 LDA 模型
    lda_model.fit(X_train, y_train)

    # 评估模型
    lda_model.evaluate(X_test, y_test)
