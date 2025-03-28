import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# ----------------------------- Naive Bayes 类 -----------------------------
class NaiveBayesModel:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.model = GaussianNB(var_smoothing=self.var_smoothing)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def visualize_decision_boundary(self, X, y):
        # 只使用前两个特征进行可视化
        X = self.scaler.transform(X[:, :2])
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)
        plt.title("Naive Bayes Decision Boundary (First 2 Features)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 只取前两个特征可视化
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb_model = NaiveBayesModel()
    nb_model.fit(X_train, y_train)
    nb_model.evaluate(X_test, y_test)

    nb_model.visualize_decision_boundary(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
