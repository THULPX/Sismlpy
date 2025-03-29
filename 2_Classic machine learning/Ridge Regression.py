import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------- Ridge Regression 算法 -----------------------------

# 介绍：
# Ridge回归是一种带有L2正则化项的线性回归方法。
# 与普通线性回归不同，Ridge回归在最小化均方误差的同时，加入了正则化项，该项旨在避免模型对数据中的噪声过拟合。
# 通过增加一个对回归系数的惩罚项，Ridge回归在训练时会控制模型的复杂度，从而提高模型的泛化能力。
# 正则化强度由超参数alpha控制，alpha越大，正则化的影响越强。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量（标签），形状为 (n_samples,)。
# 输出：
# - Ridge回归模型的预测结果。

# 算法步骤：
# 1. 假设输入特征与目标变量之间存在线性关系，构建线性模型：y = X * w + b。
# 2. 使用L2正则化，对模型参数w加入惩罚项：L2正则化是系数w的平方和乘以正则化超参数alpha。
# 3. 通过最小化目标函数来优化模型参数：L(w) = ||y - Xw||² + alpha * ||w||²。
# 4. 对新样本进行预测，得到目标变量的估计值。

# 主要参数：
# - alpha: 正则化强度，越大正则化的作用越强，默认为1.0。
# - fit_intercept: 是否计算截距项b，默认为True。
# - normalize: 是否在回归之前对数据进行标准化，默认为False。
# - solver: 用于计算的优化算法，默认为'auto'，可以选择'lbfgs', 'saga', 'cholesky'等。

class RidgeRegressionModel:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Model Mean Squared Error (MSE): {mse:.2f}")
        print(f"Model R-squared (R²): {r2:.2f}")

        # 可视化：预测值 vs 真实值
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.show()

        # 可视化：残差分布
        residuals = y_test - predictions
        plt.figure(figsize=(10, 5))
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residual Histogram")
        plt.show()

        # 可视化：特征重要性
        plt.figure(figsize=(12, 6))
        feature_importance = np.abs(self.model.coef_)
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Magnitude")
        plt.title("Feature Importance in Ridge Regression")
        plt.show()

        return mse, r2


if __name__ == "__main__":
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ridge_model = RidgeRegressionModel(alpha=1.0, fit_intercept=True)
    ridge_model.fit(X_train, y_train)
    ridge_model.evaluate(X_test, y_test)