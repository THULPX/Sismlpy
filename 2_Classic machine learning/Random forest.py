import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, target_names=None):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # 可视化特征重要性
        self.plot_feature_importance()

        # 可视化PCA预测 vs 真实标签
        self.plot_pca_results(X_test, y_test, predictions)

        # 可视化混淆矩阵
        if target_names is not None:
            self.plot_confusion_matrix(y_test, predictions, target_names)

        return accuracy

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [f"Feature {i}" for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_pca_results(self, X_test, y_test, y_pred):
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_test)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', marker='o', alpha=0.5, label='Predicted')
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_test, cmap='coolwarm', marker='x', alpha=0.5, label='True')
        plt.title("PCA Projection of Predictions vs True Labels")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()


# ----------------------------- 示例：使用鸢尾花数据集进行训练和评估 -----------------------------

if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建随机森林模型
    random_forest = RandomForestModel(n_estimators=100, max_depth=5)

    # 训练并评估模型（带可视化）
    random_forest.fit(X_train, y_train)
    random_forest.evaluate(X_test, y_test, target_names=iris.target_names)
