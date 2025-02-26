import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- Lasso Regression 算法 -----------------------------

# 介绍：
# Lasso回归（Least Absolute Shrinkage and Selection Operator）是一种带有L1正则化项的线性回归方法。Lasso回归通过加入L1正则化项来控制模型复杂度，
# 并且能够自动进行特征选择。L1正则化的作用是惩罚回归系数的绝对值，从而使一些系数变为零。通过这种方式，Lasso回归在回归分析中可以自动选择对预测有影响的特征，
# 从而简化模型并减少过拟合。

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
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000):
        """
        初始化 Lasso 回归模型。

        :param alpha: 正则化强度，默认为1.0。
        :param fit_intercept: 是否计算截距项b，默认为True。
        :param normalize: 是否对数据进行标准化，默认为False。
        :param max_iter: 最大迭代次数，默认为1000。
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Lasso 回归模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 Lasso 回归模型
        self.model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, normalize=self.normalize,
                           max_iter=self.max_iter)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Lasso 回归模型进行预测。

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
        :param y_test: 测试数据的目标变量。
        :return: 模型的评估指标，如均方误差(MSE)和R²。
        """
        predictions = self.predict(X_test)

        # 计算均方误差(MSE)
        mse = mean_squared_error(y_test, predictions)

        # 计算R²值
        r2 = r2_score(y_test, predictions)

        print(f"Model Mean Squared Error (MSE): {mse:.2f}")
        print(f"Model R-squared (R²): {r2:.2f}")
        return mse, r2


# 示例：使用波士顿房价数据集进行训练和评估
if __name__ == "__main__":
    # 加载波士顿房价数据集
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 Lasso 回归模型
    lasso_model = LassoRegressionModel(alpha=1.0, fit_intercept=True, normalize=True)

    # 训练 Lasso 回归模型
    lasso_model.fit(X_train, y_train)

    # 评估模型
    lasso_model.evaluate(X_test, y_test)
