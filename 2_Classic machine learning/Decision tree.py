import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# ----------------------------- 决策树（Decision Trees）算法 -----------------------------

# 介绍：
# 决策树是一种基于树形结构的监督学习算法，适用于分类和回归任务。
# 它通过对特征进行一系列判断（即分裂）来形成决策路径，最终根据叶子节点的类别来进行预测。
# 决策树的目标是选择最能分割数据的特征，以减少信息的不确定性（通常通过信息增益或基尼指数来评估）。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - 决策树模型的预测结果。

# 算法步骤：
# 1. 从根节点开始，对每个特征进行分裂，直到满足停止条件（如达到最大深度、样本数小于阈值等）。
# 2. 在每个节点选择最佳特征进行分裂（通过计算信息增益或基尼指数）。
# 3. 重复分裂直到叶子节点，叶子节点保存类别标签。
# 4. 使用训练好的决策树模型对新数据进行预测。

# 主要参数：
# - criterion: 划分的标准，通常有'ginni'（基尼指数）和'entropy'（信息增益）。
# - max_depth: 树的最大深度，控制模型复杂度。
# - min_samples_split: 划分内部节点所需的最小样本数。
# - min_samples_leaf: 每个叶子节点所需的最小样本数。

class DecisionTreeModel:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        初始化决策树（Decision Tree）模型。

        :param criterion: 划分标准 ('gini' 或 'entropy')。
        :param max_depth: 树的最大深度。
        :param min_samples_split: 划分内部节点所需的最小样本数。
        :param min_samples_leaf: 每个叶子节点所需的最小样本数。
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练决策树模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练决策树模型
        self.model = DecisionTreeClassifier(criterion=self.criterion,
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的决策树模型进行预测。

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

    # 创建决策树模型
    decision_tree = DecisionTreeModel(criterion='entropy', max_depth=5)

    # 训练决策树模型
    decision_tree.fit(X_train, y_train)

    # 评估模型
    decision_tree.evaluate(X_test, y_test)
