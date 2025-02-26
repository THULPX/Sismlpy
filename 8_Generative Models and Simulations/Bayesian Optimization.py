import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# ----------------------------- Bayesian Optimization -----------------------------

# 介绍：
# 贝叶斯优化是一种用于优化复杂且代价昂贵的函数的优化方法，尤其适用于高维、非凸或噪声较多的目标函数。贝叶斯优化通过构建一个概率模型（通常是高斯过程回归）来表示目标函数的分布，并使用该模型来选择下一个采样点，从而最大化目标函数的值。它主要包括两个关键步骤：
# 1. 构建代理模型（例如高斯过程回归模型），根据已有的样本数据拟合模型。
# 2. 使用采集函数（如预期改进、概率改进等）选择下一个采样点。
# 贝叶斯优化的优势在于能够高效地探索搜索空间，尤其适合函数值评估非常昂贵或不可得的情况。

# 输入输出：
# 输入：
# - f: 需要优化的目标函数。
# - bounds: 搜索空间的边界。
# - n_iter: 最大迭代次数（即最大采样次数）。
# - kernel: 高斯过程回归模型的核函数。
# 输出：
# - 最优解：目标函数的最优解。

# 算法步骤：
# 1. 初始化：选择初始的采样点并评估目标函数值。
# 2. 拟合高斯过程回归模型：基于已采样的点拟合代理模型。
# 3. 选择下一个采样点：通过采集函数选择下一个最有可能提升目标函数的点。
# 4. 评估目标函数并更新模型：计算新采样点的目标函数值，并用它来更新高斯过程模型。
# 5. 重复以上步骤，直到达到最大迭代次数。

# 主要参数：
# - f: 目标函数。
# - bounds: 搜索空间的边界。
# - n_iter: 最大迭代次数。
# - kernel: 高斯过程回归的核函数。

class BayesianOptimization:
    def __init__(self, f, bounds, n_iter=25, kernel=None):
        """
        初始化贝叶斯优化器。

        :param f: 目标函数。
        :param bounds: 搜索空间的边界。
        :param n_iter: 最大迭代次数。
        :param kernel: 高斯过程回归的核函数。
        """
        self.f = f  # 目标函数
        self.bounds = bounds  # 搜索空间边界
        self.n_iter = n_iter  # 最大迭代次数
        self.kernel = kernel if kernel is not None else C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))  # 核函数
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)  # 高斯过程回归模型
        self.X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (5, bounds.shape[0]))  # 初始采样点
        self.Y_sample = np.array([self.f(x) for x in self.X_sample])  # 初始采样点的目标函数值

    def acquisition_function(self, X, xi=0.01):
        """
        采集函数：使用预期改进作为采集函数。
        :param X: 采样点。
        :param xi: 探索参数，控制探索与利用的平衡。
        :return: 每个采样点的采集函数值。
        """
        mu, sigma = self.gpr.predict(X, return_std=True)
        mu_sample_opt = np.max(self.Y_sample)

        # 预期改进（Expected Improvement, EI）
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def optimize(self):
        """
        执行贝叶斯优化，返回最优解。
        """
        for i in range(self.n_iter):
            # 使用高斯过程回归模型拟合现有的数据
            self.gpr.fit(self.X_sample, self.Y_sample)

            # 使用采集函数选择下一个采样点
            X_next = self.pick_next_sample()

            # 评估目标函数并更新样本
            Y_next = self.f(X_next)
            self.X_sample = np.vstack((self.X_sample, X_next))
            self.Y_sample = np.append(self.Y_sample, Y_next)

        # 返回最优的解
        best_index = np.argmin(self.Y_sample)
        return self.X_sample[best_index], self.Y_sample[best_index]

    def pick_next_sample(self):
        """
        选择下一个采样点，最大化采集函数。
        """
        X_tries = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (1000, self.bounds.shape[0]))
        ei = np.array([self.acquisition_function(x.reshape(1, -1)) for x in X_tries])

        # 返回使采集函数值最大的采样点
        return X_tries[np.argmax(ei)]


# ----------------------------- 示例：使用贝叶斯优化 -----------------------------

# 定义目标函数：例如目标函数为Rastrigin函数
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# 设置搜索空间边界：-5到5
bounds = np.array([[-5, 5]] * 2)

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(f=rastrigin, bounds=bounds, n_iter=30)

# 执行优化
best_x, best_y = optimizer.optimize()

print(f"最优解: {best_x}")
print(f"最优目标函数值: {best_y}")
