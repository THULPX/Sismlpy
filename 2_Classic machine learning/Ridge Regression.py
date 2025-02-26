import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- Ridge Regression 算法 -----------------------------

# 介绍：
# Ridge回归是一种带有L2正则化项的线性回归方法。与普通线性回归不同，Ridge回归在最小化均方误差的同时，加入了正则化项，
# 该项旨在避免模型对数据中的噪声过拟合。通过增加一个对回归系数的惩罚项，Ridge回归在训练时会控制模型的复杂度，
# 从而提高模型的泛化能力。正则化强度由超参数alpha控制，alpha越大，正则化的影响越强。

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
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False):
        """
        初始化 Ridge 回归模型。

        :param alpha: 正则化强度，默认为1.0。
        :param fit_intercept: 是否计算截距项b，默认为True。
        :param normalize: 是否对数据进行标准化，默认为False。
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Ridge 回归模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # 初始化并训练 Ridge 回归模型
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, normalize=self.normalize)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Ridge 回归模型进行预测。

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

    # 创建 Ridge 回归模型
    ridge_model = RidgeRegressionModel(alpha=1.0, fit_intercept=True, normalize=True)

    # 训练 Ridge 回归模型
    ridge_model.fit(X_train, y_train)

    # 评估模型
    ridge_model.evaluate(X_test, y_test)
