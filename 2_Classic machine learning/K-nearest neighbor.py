import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# ----------------------------- K-Nearest Neighbors (KNN) 算法 -----------------------------

class KNNModel:
    def __init__(self, n_neighbors=5, metric='minkowski'):
        """
        初始化 K-Nearest Neighbors 模型。

        :param n_neighbors: K值，即选择的邻居数。
        :param metric: 距离度量方式，默认为'minkowski'，也可以选择其他方式如'euclidean'。
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 KNN 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 KNN 模型
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 KNN 模型进行预测。

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

    def plot_decision_boundary(self, X, y):
        """
        绘制决策边界。

        :param X: 数据的特征。
        :param y: 数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 创建网格点
        h = .02  # 网格步长
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 预测每个网格点的类别
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # 绘制训练数据点
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"KNN (k={self.n_neighbors}) Decision Boundary")
        plt.show()

# 示例：使用鸢尾花数据集进行训练和评估
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 为了可视化，只使用前两个特征
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 KNN 模型
    knn_model = KNNModel(n_neighbors=7, metric='minkowski')

    # 训练 KNN 模型
    knn_model.fit(X_train, y_train)

    # 评估模型
    accuracy = knn_model.evaluate(X_test, y_test)

    # 绘制决策边界
    knn_model.plot_decision_boundary(X_train, y_train)

    # 可视化准确率随 K 值变化的曲线
    k_values = range(1, 31)
    accuracies = []
    for k in k_values:
        knn_model = KNNModel(n_neighbors=k, metric='minkowski')
        knn_model.fit(X_train, y_train)
        accuracy = knn_model.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    plt.figure()
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. K Value')
    plt.show()