import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- Gaussian Processes (高斯过程) 算法 -----------------------------

# 介绍：
# 高斯过程回归（Gaussian Process Regression，GPR）是一种基于贝叶斯理论的非参数回归方法。与传统的回归方法不同，高斯过程回归不对模型进行假设（例如线性关系），
# 而是通过样本点之间的协方差来进行建模。在高斯过程中，每一个训练样本都看作是从某个潜在的高斯分布中采样而来，因此它能够在给定训练数据的情况下，输出一个关于预测值的不确定性估计。
# 高斯过程回归特别适用于当数据量较小并且存在较强非线性关系的场景。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标变量，形状为 (n_samples,)。
# 输出：
# - 高斯过程回归模型的预测结果。

# 算法步骤：
# 1. 定义一个高斯过程模型，其假设所有的函数值是从一个联合高斯分布中采样得到的。
# 2. 根据训练数据点，计算训练数据之间的协方差矩阵。
# 3. 使用最大似然估计优化高斯过程的超参数。
# 4. 在新的测试数据点上进行预测，输出预测的均值和方差。
# 5. 评估模型性能，通过比较预测值与真实值来计算评估指标。

# 主要参数：
# - kernel: 高斯过程的核函数，默认为 RBF 核。
# - alpha: 噪声的方差，默认为1e-10。
# - n_restarts_optimizer: 优化器的重启次数，默认为10。

class GaussianProcessModel:
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=10):
        """
        初始化高斯过程回归模型。

        :param kernel: 高斯过程核函数，默认为RBF核。
        :param alpha: 噪声的方差，默认为1e-10。
        :param n_restarts_optimizer: 优化器重启次数，默认为10。
        """
        # 默认使用RBF核
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练高斯过程回归模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 初始化并训练高斯过程回归模型
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha,
                                              n_restarts_optimizer=self.n_restarts_optimizer)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的高斯过程回归模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果及其方差。
        """
        predictions, std_devs = self.model.predict(X_test, return_std=True)
        return predictions, std_devs

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量。
        :return: 模型的评估指标，如均方误差(MSE)和R²。
        """
        predictions, _ = self.predict(X_test)

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

    # 创建高斯过程回归模型
    gp_model = GaussianProcessModel()

    # 训练高斯过程回归模型
    gp_model.fit(X_train, y_train)

    # 评估模型
    gp_model.evaluate(X_test, y_test)
