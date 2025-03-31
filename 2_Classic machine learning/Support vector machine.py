import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------- 支持向量机（SVM）算法 -----------------------------

class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        """
        只适用于二维特征
        """
        X = self.scaler.transform(X)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def plot_learning_curve(self, X, y, C_values):
        train_acc = []
        test_acc = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for C in C_values:
            self.C = C
            self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.model.fit(X_train_scaled, y_train)

            train_acc.append(self.model.score(X_train_scaled, y_train))
            test_acc.append(self.model.score(X_test_scaled, y_test))

        plt.figure(figsize=(8, 6))
        plt.plot(C_values, train_acc, label='Training Accuracy', marker='o')
        plt.plot(C_values, test_acc, label='Testing Accuracy', marker='x')
        plt.xscale('log')
        plt.xlabel("C (log scale)")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve: Accuracy vs C")
        plt.legend()
        plt.grid(True)
        plt.show()


# 示例：使用鸢尾花数据集进行训练和可视化
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 只取前两个特征便于可视化
    y = iris.target

    svm = SVMModel(kernel='rbf', C=1.0, gamma='scale')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    svm.fit(X_train, y_train)
    svm.evaluate(X_test, y_test)

    # 可视化决策边界
    svm.plot_decision_boundary(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))

    # 学习曲线：不同 C 值的表现
    C_range = np.logspace(-2, 2, 10)
    svm.plot_learning_curve(X, y, C_range)
