import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ----------------------------- 随机森林（Random Forests）算法 -----------------------------

# 介绍：
# 随机森林（Random Forest）是一种集成学习算法，通过构建多个决策树并结合它们的预测结果来提高模型的准确性和鲁棒性。
# 随机森林利用“袋装法”（Bootstrap Aggregating，Bagging）构建多棵决策树，并通过投票或平均来做出最终的预测结果。
# 它通过随机选择特征子集来训练每棵树，从而减少模型的过拟合，并增强预测的稳定性。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - 随机森林模型的预测结果。

# 算法步骤：
# 1. 从训练数据集中通过有放回抽样方法构建多个训练子集。
# 2. 对每个训练子集，训练一个决策树模型。
# 3. 在每个节点分裂时，随机选择特征子集而不是所有特征。
# 4. 对于分类任务，所有树的投票决定最终分类结果；对于回归任务，取所有树的平均值。
# 5. 使用训练好的随机森林模型对新数据进行预测。

# 主要参数：
# - n_estimators: 森林中树的数量。
# - max_depth: 每棵树的最大深度。
# - min_samples_split: 内部节点再划分所需的最小样本数。
# - min_samples_leaf: 叶子节点的最小样本数。
# - max_features: 每次分裂时选择的最大特征数。

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        初始化随机森林（Random Forest）模型。

        :param n_estimators: 森林中树的数量。
        :param max_depth: 每棵树的最大深度。
        :param min_samples_split: 内部节点再划分所需的最小样本数。
        :param min_samples_leaf: 叶子节点的最小样本数。
        :param max_features: 每次分裂时选择的最大特征数。
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练随机森林模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练随机森林模型
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            max_features=self.max_features)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的随机森林模型进行预测。

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

    # 创建随机森林模型
    random_forest = RandomForestModel(n_estimators=100, max_depth=5)

    # 训练随机森林模型
    random_forest.fit(X_train, y_train)

    # 评估模型
    random_forest.evaluate(X_test, y_test)
