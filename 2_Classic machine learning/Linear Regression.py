import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- Linear Regression 算法 -----------------------------

# 介绍：
# 线性回归是一种用于回归任务的统计学习方法，通过拟合一个线性模型来预测目标变量（即因变量）。
# 它的基本假设是，因变量与一个或多个自变量之间存在线性关系。
# 最常用的线性回归方法是最小二乘法，它通过最小化预测值和真实值之间的平方误差来优化模型参数。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)。
# 输出：
# - 线性回归模型的预测结果。

# 算法步骤：
# 1. 假设输入特征和目标变量之间存在线性关系，构建线性模型：y = X * w + b。
# 2. 使用训练数据集，通过最小化均方误差（MSE）来计算模型参数 w 和 b。
# 3. 通过计算 MSE 或者 R²等评估指标，评估模型的拟合效果。
# 4. 使用训练好的模型对新样本进行预测。

# 主要参数：
# - fit_intercept: 是否计算截距项b，默认为True。
# - normalize: 是否在回归之前对数据进行标准化，默认为False。
# - n_jobs: 用于并行计算的工作线程数，默认为1。


class LinearRegressionModel:
    def __init__(self, fit_intercept=True, normalize=False):
        """
        初始化线性回归模型。

        :param fit_intercept: 是否计算截距项b，默认为True。
        :param normalize: 是否对数据进行标准化，默认为False。
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.model = None
        self.scaler = StandardScaler() if normalize else None

    def fit(self, X_train, y_train):
        """
        训练线性回归模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的线性回归模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果。
        """
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量。
        :return: 模型的评估指标，如均方误差(MSE)和R²。
        """
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Model Mean Squared Error (MSE): {mse:.2f}")
        print(f"Model R-squared (R²): {r2:.2f}")
        return mse, r2


def plot_predictions(y_true, y_pred):
    """
    可视化预测值与真实值的散点图。

    :param y_true: 真实值。
    :param y_pred: 预测值。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Linear Regression: Predictions vs True Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------- 示例运行 -----------------------------
if __name__ == "__main__":
    # 加载 California 房价数据集
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 创建线性回归模型
    lr_model = LinearRegressionModel(fit_intercept=True, normalize=True)

    # 训练模型
    lr_model.fit(X_train, y_train)

    # 评估模型
    lr_model.evaluate(X_test, y_test)

    # 可视化预测效果
    y_pred = lr_model.predict(X_test)
    plot_predictions(y_test, y_pred)