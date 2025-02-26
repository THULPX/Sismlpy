import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ----------------------------- 梯度提升树（Gradient Boosting Machines, GBM）算法 -----------------------------

# 介绍：
# 梯度提升树（Gradient Boosting Machines, GBM）是一种集成学习算法，通过构建多个决策树来提高模型的性能。
# GBM是基于梯度提升（Gradient Boosting）的方法，逐步通过拟合残差（即预测值与真实值之间的差异）来增强模型。
# 每个新的决策树都是在前一个树的基础上优化的，从而使得整个模型能够更好地拟合数据。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - 梯度提升模型的预测结果。

# 算法步骤：
# 1. 初始化一个简单的预测模型（通常是常数值预测）。
# 2. 计算当前预测与真实标签之间的残差。
# 3. 基于残差训练一个新的决策树。
# 4. 将新决策树的预测结果加权到当前模型中。
# 5. 重复步骤2-4，直到达到预设的迭代次数或误差收敛。
# 6. 使用训练好的GBM模型对新数据进行预测。

# 主要参数：
# - n_estimators: 基学习器（树）的数量。
# - learning_rate: 每个模型的贡献缩放因子，控制学习步长。
# - max_depth: 每棵树的最大深度，控制树的复杂度。
# - min_samples_split: 内部节点再划分所需的最小样本数。
# - min_samples_leaf: 叶子节点的最小样本数。
# - subsample: 每次迭代时使用的样本比例。

class GBMModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 subsample=1.0):
        """
        初始化梯度提升树（Gradient Boosting Machine, GBM）模型。

        :param n_estimators: 基学习器（树）的数量。
        :param learning_rate: 每个模型的贡献缩放因子，控制学习步长。
        :param max_depth: 每棵树的最大深度。
        :param min_samples_split: 内部节点再划分所需的最小样本数。
        :param min_samples_leaf: 叶子节点的最小样本数。
        :param subsample: 每次迭代时使用的样本比例。
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练梯度提升树模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练梯度提升树模型
        self.model = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                learning_rate=self.learning_rate,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                subsample=self.subsample)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的梯度提升树模型进行预测。

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
        accuracy = np.mean(predictions == y_test)
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

    # 创建梯度提升树模型
    gbm = GBMModel(n_estimators=100, learning_rate=0.1, max_depth=3)

    # 训练梯度提升树模型
    gbm.fit(X_train, y_train)

    # 评估模型
    gbm.evaluate(X_test, y_test)
