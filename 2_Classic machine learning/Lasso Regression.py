import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- Lasso Regression 算法 -----------------------------

# 介绍：
# Lasso回归（Least Absolute Shrinkage and Selection Operator）是一种带有L1正则化项的线性回归方法。
# Lasso回归通过加入L1正则化项来控制模型复杂度，并且能够自动进行特征选择。
# L1正则化的作用是惩罚回归系数的绝对值，从而使一些系数变为零。
# 通过这种方式，Lasso回归在回归分析中可以自动选择对预测有影响的特征，从而简化模型并减少过拟合。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)。
# 输出：
# - Lasso回归模型的预测结果。

# 算法步骤：
# 1. 假设输入特征与目标变量之间存在线性关系，构建线性模型：y = X * w + b。
# 2. 使用L1正则化，对模型参数w加入惩罚项：L1正则化是系数w的绝对值之和乘以正则化超参数alpha。
# 3. 通过最小化目标函数来优化模型参数：L(w) = ||y - Xw||² + alpha * ||w||₁。
# 4. 对新样本进行预测，得到目标变量的估计值。

# 主要参数：
# - alpha: 正则化强度，默认为1.0。alpha越大，正则化强度越大，模型越简单。
# - fit_intercept: 是否计算截距项b，默认为True。
# - normalize: 是否在回归之前对数据进行标准化，默认为False。
# - max_iter: 最大迭代次数，默认为1000。


class LassoRegressionModel:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                           max_iter=self.max_iter, warm_start=True)
        self.model.fit(X_train_scaled, y_train)
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Model Mean Squared Error (MSE): {mse:.2f}")
        print(f"Model R-squared (R²): {r2:.2f}")
        return predictions, mse, r2

    def plot_prediction_vs_actual(self, y_test, predictions):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Lasso Regression: Predicted vs Actual")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_feature_weights(self, feature_names):
        plt.figure(figsize=(10, 5))
        plt.bar(feature_names, self.model.coef_)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.title("Lasso Feature Weights")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 示例：使用 California housing 数据集
if __name__ == "__main__":
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lasso_model = LassoRegressionModel(alpha=0.01, fit_intercept=True)
    lasso_model.fit(X_train, y_train)

    predictions, mse, r2 = lasso_model.evaluate(X_test, y_test)
    lasso_model.plot_prediction_vs_actual(y_test, predictions)
    lasso_model.plot_feature_weights(feature_names)
