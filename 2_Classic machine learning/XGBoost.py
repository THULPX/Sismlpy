import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------- XGBoost 算法 -----------------------------

# 介绍：
# XGBoost（Extreme Gradient Boosting）是一个高效的梯度提升算法，通常用于结构化数据的分类和回归任务。
# 它通过优化传统的梯度提升（GBM）方法，在计算效率和预测性能上取得了显著的提升。
# XGBoost采用了二阶导数信息，使得每次迭代更加精确，并且具有强大的正则化能力，防止过拟合。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - XGBoost 模型的预测结果。

# 算法步骤：
# 1. 初始化模型，设置基学习器的数量（n_estimators）和学习率（learning_rate）。
# 2. 每次迭代训练一个决策树，并通过梯度提升优化残差。
# 3. 每棵树的输出加权贡献到最终的预测结果中。
# 4. 使用训练好的XGBoost模型对新数据进行预测。

# 主要参数：
# - n_estimators: 基学习器（树）的数量。
# - learning_rate: 每个树的贡献缩放因子，控制学习步长。
# - max_depth: 每棵树的最大深度。
# - min_child_weight: 叶子节点的最小样本权重和。
# - subsample: 每次迭代时使用的样本比例。
# - colsample_bytree: 每棵树训练时使用的特征比例。
# - gamma: 控制每个分裂的最小损失函数减少。

class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, gamma=0):
        """
        初始化 XGBoost 模型。

        :param n_estimators: 基学习器（树）的数量。
        :param learning_rate: 每棵树的贡献缩放因子。
        :param max_depth: 每棵树的最大深度。
        :param min_child_weight: 叶子节点的最小样本权重和。
        :param subsample: 每次迭代时使用的样本比例。
        :param colsample_bytree: 每棵树训练时使用的特征比例。
        :param gamma: 控制每个分裂的最小损失函数减少。
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 XGBoost 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 XGBoost 模型
        self.model = xgb.XGBClassifier(n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate,
                                       max_depth=self.max_depth,
                                       min_child_weight=self.min_child_weight,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       gamma=self.gamma,
                                       objective='multi:softmax',  # 对多分类问题使用 softmax
                                       num_class=3)  # 适应鸢尾花数据集（3类）
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 XGBoost 模型进行预测。

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

    # 创建 XGBoost 模型
    xgboost_model = XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=3)

    # 训练 XGBoost 模型
    xgboost_model.fit(X_train, y_train)

    # 评估模型
    xgboost_model.evaluate(X_test, y_test)
