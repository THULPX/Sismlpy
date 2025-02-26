import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------- AdaBoost 算法 -----------------------------

# 介绍：
# AdaBoost（Adaptive Boosting）是一种提升方法，它通过组合多个弱分类器（通常是浅层决策树）来构建一个强分类器。
# 在每一轮迭代中，AdaBoost会调整每个样本的权重，重点关注那些被前一轮分类器错误分类的样本。
# 通过不断优化模型的错误，最终实现强大的预测能力。AdaBoost能有效减少过拟合，尤其在小数据集上表现良好。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - AdaBoost 模型的预测结果。

# 算法步骤：
# 1. 初始化训练数据集的权重，通常将所有样本的权重设置为相等。
# 2. 在每次迭代中，使用当前权重训练一个弱分类器。
# 3. 根据弱分类器的表现（错误率）调整样本的权重：分类错误的样本权重增加，分类正确的样本权重减少。
# 4. 根据弱分类器的错误率，计算其权重，最终将多个弱分类器组合成一个强分类器。
# 5. 重复步骤2-4，直到达到指定的迭代次数或模型的表现不再提高。
# 6. 使用训练好的AdaBoost模型对新数据进行预测。

# 主要参数：
# - n_estimators: 弱分类器的数量。
# - learning_rate: 每个弱分类器的权重缩放因子。
# - base_estimator: 基学习器，通常是浅层决策树。
# - algorithm: 使用的提升算法（SAMME 或 SAMME.R）。

class AdaBoostModel:
    def __init__(self, n_estimators=50, learning_rate=1.0, base_estimator=None):
        """
        初始化 AdaBoost 模型。

        :param n_estimators: 弱分类器的数量。
        :param learning_rate: 每个弱分类器的权重缩放因子。
        :param base_estimator: 基学习器，通常是浅层决策树。
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 AdaBoost 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 AdaBoost 模型
        self.model = AdaBoostClassifier(base_estimator=self.base_estimator,
                                        n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 AdaBoost 模型进行预测。

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

    # 创建 AdaBoost 模型
    adaboost_model = AdaBoostModel(n_estimators=50, learning_rate=1.0)

    # 训练 AdaBoost 模型
    adaboost_model.fit(X_train, y_train)

    # 评估模型
    adaboost_model.evaluate(X_test, y_test)
