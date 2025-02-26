import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----------------------------- 支持向量机（SVM）算法 -----------------------------

# 介绍：
# 支持向量机（Support Vector Machine, SVM）是一种监督学习算法，广泛应用于分类和回归问题。
# SVM通过在高维空间中寻找一个最优超平面来最大化类之间的间隔，从而进行分类。它利用支持向量（即距离分类边界最近的样本点）来建立分类模型。
# SVM特别适用于高维特征空间和复杂的非线性决策边界。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - SVM模型的预测结果。

# 算法步骤：
# 1. 从训练数据中构建一个最大间隔超平面（线性或非线性）。
# 2. 找到支持向量，即距离超平面最近的样本。
# 3. 使用优化算法求解最大化间隔的超平面。
# 4. 用训练好的SVM模型对新数据进行分类或回归预测。

# 主要参数：
# - C: 惩罚参数，控制对误分类的容忍度。
# - kernel: 核函数类型（'linear', 'poly', 'rbf', 'sigmoid'等）。
# - gamma: 核函数的参数，控制样本的影响范围。

class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        初始化支持向量机（SVM）模型。

        :param kernel: 核函数类型（'linear', 'poly', 'rbf', 'sigmoid'等）。
        :param C: 惩罚参数，控制对误分类的容忍度。
        :param gamma: 核函数的参数。
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练支持向量机模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练SVM模型
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的SVM模型进行预测。

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

    # 创建SVM模型
    svm = SVMModel(kernel='rbf', C=1.0, gamma='scale')

    # 训练SVM模型
    svm.fit(X_train, y_train)

    # 评估模型
    svm.evaluate(X_test, y_test)
