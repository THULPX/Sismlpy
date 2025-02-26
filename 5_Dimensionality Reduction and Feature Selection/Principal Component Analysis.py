import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ----------------------------- Principal Component Analysis (PCA) 算法 -----------------------------

# 介绍：
# 主成分分析（Principal Component Analysis, PCA）是一种降维技术，通过线性变换将数据从高维空间映射到低维空间，同时尽可能保留数据的方差（信息量）。PCA 通过计算数据的协方差矩阵并对其进行特征值分解，得到主成分（新的轴）。
# PCA 适用于数据的降维和去噪，它能够有效地减少特征维度，从而提高机器学习算法的效率和性能。PCA 常用于数据可视化、压缩、去噪和特征选择。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# 输出：
# - 降维后的数据，形状为 (n_samples, n_components)，其中 n_components 是降维后的维度数。

# 算法步骤：
# 1. 对数据进行标准化，确保每个特征的均值为 0，标准差为 1。
# 2. 计算数据的协方差矩阵，描述数据各特征之间的相关性。
# 3. 对协方差矩阵进行特征值分解，得到特征向量和对应的特征值。
# 4. 按照特征值的大小选择前 k 个特征向量，形成主成分。
# 5. 使用选择的主成分将原始数据投影到低维空间。

# 主要参数：
# - n_components: 降维后的维度数，默认为 2。
# - whiten: 是否对数据进行白化，默认为 False。
# - svd_solver: 奇异值分解求解器，默认为 'auto'，其他选项包括 'full', 'arpack', 'randomized'。

class PCAModel:
    def __init__(self, n_components=2, whiten=False, svd_solver='auto'):
        """
        初始化 PCA 模型。

        :param n_components: 降维后的维度数，默认为 2。
        :param whiten: 是否对数据进行白化，默认为 False。
        :param svd_solver: 奇异值分解求解器，默认为 'auto'。
        """
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.model = None

    def fit(self, X_train):
        """
        训练 PCA 模型。

        :param X_train: 训练数据的特征。
        """
        # 初始化并训练 PCA 模型
        self.model = PCA(n_components=self.n_components, whiten=self.whiten, svd_solver=self.svd_solver)
        self.model.fit(X_train)
        print("Model trained successfully.")

    def transform(self, X):
        """
        使用训练好的 PCA 模型进行数据降维。

        :param X: 输入数据的特征。
        :return: 降维后的数据。
        """
        transformed_data = self.model.transform(X)
        return transformed_data

    def fit_transform(self, X_train):
        """
        训练 PCA 模型并对数据进行降维。

        :param X_train: 训练数据的特征。
        :return: 降维后的数据。
        """
        transformed_data = self.model.fit_transform(X_train)
        return transformed_data

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能（通过降维后的数据进行分类评估）。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量（标签）。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        transformed_data = self.transform(X_test)

        # 使用降维后的数据进行简单分类（例如 KNN）评估
        from sklearn.neighbors import KNeighborsClassifier

        # 训练 KNN 分类器
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(transformed_data, y_test)

        # 预测
        predictions = knn.predict(transformed_data)

        # 评估准确率
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

    # 创建 PCA 模型
    pca_model = PCAModel(n_components=2)

    # 训练 PCA 模型并进行降维
    X_train_pca = pca_model.fit_transform(X_train)

    # 评估模型
    pca_model.evaluate(X_test, y_test)
