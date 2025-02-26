import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


# ----------------------------- Newton's Method -----------------------------

# 介绍：
# 牛顿法（Newton's Method）是一种迭代优化方法，主要用于求解方程的根或者寻找函数的最小值。它通过利用目标函数的一阶导数（梯度）和二阶导数（海森矩阵）来加速求解过程。牛顿法常用于无约束优化问题，能够在目标函数具有良好条件下快速收敛。
# 在机器学习中，牛顿法可以用于优化损失函数（如最小二乘法），尤其是在处理大规模数据时。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - gradient: 目标函数的梯度（导数）。
# - hessian: 目标函数的海森矩阵（二阶导数矩阵）。
# - initial_guess: 初始解。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。
# 输出：
# - 最优解及其目标函数值。

# 算法步骤：
# 1. 初始化解为初始猜测值。
# 2. 计算目标函数的梯度和海森矩阵。
# 3. 更新解：利用牛顿法的更新公式 x(k+1) = x(k) - H^-1 * g，其中 H^-1 为海森矩阵的逆，g 为梯度。
# 4. 判断收敛：如果梯度的范数小于容忍度，停止迭代。
# 5. 重复步骤 2-4，直到达到最大迭代次数或满足收敛条件。

# 主要参数：
# - cost_function: 目标函数。
# - gradient: 目标函数的梯度。
# - hessian: 目标函数的海森矩阵。
# - initial_guess: 初始猜测值。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。

def newtons_method(cost_function, gradient, hessian, initial_guess, max_iter=1000, tolerance=1e-6):
    """
    使用牛顿法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param gradient: 目标函数的梯度（导数）。
    :param hessian: 目标函数的海森矩阵（二阶导数矩阵）。
    :param initial_guess: 初始解。
    :param max_iter: 最大迭代次数。
    :param tolerance: 收敛容忍度。
    :return: 最优解及其目标函数值。
    """
    x = initial_guess
    for i in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)

        # 计算更新步长
        try:
            hess_inv = np.linalg.inv(hess)  # 求海森矩阵的逆
            x_new = x - np.dot(hess_inv, grad)  # 牛顿法更新公式
        except np.linalg.LinAlgError:
            print("海森矩阵不可逆，算法无法继续执行。")
            return x, cost_function(x)

        # 检查收敛条件
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            return x_new, cost_function(x_new)

        x = x_new

    return x, cost_function(x)


# 示例：使用牛顿法最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # 目标函数：均方误差
    def cost_function(w):
        predictions = X_train.dot(w)
        return mean_squared_error(y_train, predictions)


    # 目标函数的梯度（偏导数）
    def gradient(w):
        predictions = X_train.dot(w)
        grad = -2 * X_train.T.dot(y_train - predictions) / len(y_train)
        return grad


    # 目标函数的海森矩阵（二阶偏导数）
    def hessian(w):
        hess = 2 * X_train.T.dot(X_train) / len(y_train)
        return hess


    # 初始猜测值
    initial_guess = np.zeros((X_train.shape[1], 1))

    # 使用牛顿法优化
    best_solution, best_cost = newtons_method(cost_function, gradient, hessian, initial_guess)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
