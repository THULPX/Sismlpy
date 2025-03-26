import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# ----------------------------- Logistic Regression 算法 -----------------------------

# 介绍：
# 逻辑回归（Logistic Regression）是一种广泛使用的分类模型，通常用于二分类问题。
# 它通过使用sigmoid函数（逻辑函数）将线性回归模型的输出映射到[0,1]范围内，从而输出事件发生的概率。
# 尽管名字中包含“回归”二字，逻辑回归实际上是一个分类算法，而非回归算法。
# 逻辑回归的目标是通过最大化似然函数来估计模型参数，使得预测概率与实际标签之间的误差最小。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)，类别值为0或1。
# 输出：
# - Logistic回归模型的预测结果。

# 算法步骤：
# 1. 将输入特征与目标变量之间的关系建模为：P(y=1|X) = sigmoid(Xw + b)，其中sigmoid(z) = 1 / (1 + exp(-z))。
# 2. 通过最大化对数似然函数来估计模型参数w和b，常用优化方法有梯度下降法。
# 3. 模型训练完成后，通过阈值判断将预测概率转化为分类结果（通常阈值为0.5，即当P(y=1|X) >= 0.5时预测为1，反之为0）。
# 4. 对新样本进行预测，得到分类标签。

# 主要参数：
# - C: 正则化强度的倒数，越小表示正则化越强，默认为1.0。
# - penalty: 正则化类型，选择'L2'（默认）或'L1'。
# - solver: 优化算法的选择，常用的有'liblinear', 'newton-cg', 'lbfgs', 'saga'等。
# - max_iter: 最大迭代次数，默认为100。


class LogisticRegressionModel:
    def __init__(self, C=1.0, penalty='l2', solver='liblinear', max_iter=100):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()  # 改为实例变量以复用

    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(C=self.C, penalty=self.penalty, solver=self.solver, max_iter=self.max_iter, verbose=0)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        print(f"Model Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # 可视化混淆矩阵
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Setosa", "Setosa"], yticklabels=["Not Setosa", "Setosa"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        return accuracy, conf_matrix, class_report

    def visualize_predictions(self, X, y, title="Model Predictions"):
        # 降维至2D进行可视化
        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        y_pred = self.model.predict(X_scaled)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Class 0 (True)", alpha=0.5)
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Class 1 (True)", alpha=0.5)
        plt.scatter(X_pca[y_pred != y, 0], X_pca[y_pred != y, 1], c='red', marker='x', label="Misclassified")
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # Setosa vs Not Setosa

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logistic_model = LogisticRegressionModel(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    logistic_model.fit(X_train, y_train)
    logistic_model.evaluate(X_test, y_test)

    # 可视化预测结果（训练集和测试集）
    logistic_model.visualize_predictions(X_train, y_train, title="Training Set Prediction Visualization")
    logistic_model.visualize_predictions(X_test, y_test, title="Test Set Prediction Visualization")
