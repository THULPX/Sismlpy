import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------- Naive Bayes 算法 -----------------------------

# 介绍：
# Naive Bayes 是基于贝叶斯定理的一个分类算法，适用于处理文本分类等问题。它假设特征之间是条件独立的，因此称为“朴素”。
# 朴素贝叶斯方法通过计算每个类的条件概率，来对样本进行分类。常见的变种包括高斯朴素贝叶斯（GaussianNB）、多项式朴素贝叶斯（MultinomialNB）和伯努利朴素贝叶斯（BernoulliNB）。
# 高斯朴素贝叶斯适用于特征是连续值的情况，并假设特征符合高斯分布。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - Naive Bayes 模型的预测结果。

# 算法步骤：
# 1. 根据训练数据集，计算每个类别的先验概率。
# 2. 对每个特征，计算它在每个类别下的条件概率（根据高斯分布或其他分布）。
# 3. 对于一个待分类的样本，计算它在所有类别下的后验概率，并选择后验概率最大的类别作为预测结果。

# 主要参数：
# - var_smoothing: 高斯分布的平滑参数，避免除零错误。
# - priors: 各类别的先验概率，默认为均匀分布。
# - fit_prior: 是否估计类别的先验概率，默认True。

class NaiveBayesModel:
    def __init__(self, var_smoothing=1e-9):
        """
        初始化 Naive Bayes 模型。

        :param var_smoothing: 高斯分布的平滑参数，默认为1e-9。
        """
        self.var_smoothing = var_smoothing
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Naive Bayes 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 Naive Bayes 模型
        self.model = GaussianNB(var_smoothing=self.var_smoothing)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Naive Bayes 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果。
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
        :param y_test: 测试数据的标签。
        :return: 模型准确率。
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy


# 示例：使用鸢尾花数据集进行训练和评估
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 Naive Bayes 模型
    nb_model = NaiveBayesModel(var_smoothing=1e-9)

    # 训练 Naive Bayes 模型
    nb_model.fit(X_train, y_train)

    # 评估模型
    nb_model.evaluate(X_test, y_test)
