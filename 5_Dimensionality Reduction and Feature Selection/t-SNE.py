import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------- t-SNE (t-Distributed Stochastic Neighbor Embedding) 算法 -----------------------------

# 介绍：
# t-SNE（t-分布随机邻居嵌入）是一种非线性降维技术，常用于数据可视化。t-SNE通过最小化高维空间数据点之间的相似性与低维空间数据点之间的相似性之间的差异，将数据从高维空间映射到低维空间，通常降到2维或3维。
# t-SNE 尤其适用于高维数据集的可视化，如图像、文本或基因数据集等，可以揭示数据的群聚结构和类别之间的关系。它不同于PCA，因为PCA是线性降维方法，而t-SNE是非线性的。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 降维后的数据，形状为 (n_samples, 2) 或 (n_samples, 3)，通常降到2维或者3维。

# 算法步骤：
# 1. 计算高维数据点之间的概率分布，衡量它们之间的相似度。
# 2. 将高维数据的相似度映射到低维空间，并使用t分布来估计低维空间数据点之间的相似度。
# 3. 使用梯度下降法最小化高维空间和低维空间数据点之间的相似度差异，获得最终的低维嵌入。

# 主要参数：
# - n_components: 降维后的维度数，默认为 2（通常用于可视化）。
# - perplexity: 近邻的数量，用来平衡全局与局部数据结构，默认为 30。
# - learning_rate: 学习率，控制梯度下降的步伐，默认为 200。
# - n_iter: 迭代次数，默认为 1000。
# - random_state: 随机种子，默认为 None。

class TSNEModel:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=None):
        """
        初始化 t-SNE 模型。

        :param n_components: 降维后的维度数，默认为 2。
        :param perplexity: 近邻的数量，默认为 30。
        :param learning_rate: 学习率，默认为 200。
        :param n_iter: 迭代次数，默认为 1000。
        :param random_state: 随机种子，默认为 None。
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X_train):
        """
        训练 t-SNE 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 t-SNE 模型
        self.model = TSNE(n_components=self.n_components, perplexity=self.perplexity,
                          learning_rate=self.learning_rate, n_iter=self.n_iter, random_state=self.random_state)
        self.transformed_data = self.model.fit_transform(X_train)
        print("Model trained successfully.")

    def transform(self, X):
        """
        使用训练好的 t-SNE 模型进行数据降维。

        :param X: 输入数据的特征。
        :return: 降维后的数据。
        """
        transformed_data = self.model.fit_transform(X)
        return transformed_data

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过降维后的数据进行分类评估）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标标签。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        transformed_data = self.transform(X_test)

        # 使用降维后的数据进行简单分类（例如 KNN）评估
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(transformed_data, y_test)

        # 预测
        predictions = knn.predict(transformed_data)

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

    # 创建 t-SNE 模型
    tsne_model = TSNEModel(n_components=2)

    # 训练 t-SNE 模型并进行降维
    tsne_model.fit(X_train)

    # 评估模型
    tsne_model.evaluate(X_test, y_test)
