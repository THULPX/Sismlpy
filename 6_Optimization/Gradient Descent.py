import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


# ----------------------------- Gradient Descent -----------------------------

# 介绍：
# 梯度下降（Gradient Descent）是一种用于优化的迭代算法，常用于最小化目标函数。它通过沿着目标函数的梯度方向更新参数，逐步接近最小值。梯度下降的基本思想是，通过计算目标函数的梯度，判断每次更新应朝哪个方向前进。梯度下降可用于许多机器学习算法中，如线性回归、逻辑回归、神经网络等。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - gradient: 目标函数的梯度（导数）。
# - initial_guess: 初始解。
# - learning_rate: 学习率，控制每次更新的步长。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。
# 输出：
# - 最优解及其目标函数值。

# 算法步骤：
# 1. 初始化解为初始猜测值。
# 2. 计算目标函数的梯度。
# 3. 更新解：通过梯度方向调整解的值，公式为 x(k+1) = x(k) - alpha * grad，其中 alpha 为学习率，grad 为梯度。
# 4. 判断收敛：如果梯度的范数小于容忍度，停止迭代。
# 5. 重复步骤 2-4，直到达到最大迭代次数或满足收敛条件。

# 主要参数：
# - cost_function: 目标函数。
# - gradient: 目标函数的梯度。
# - initial_guess: 初始猜测值。
# - learning_rate: 学习率。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。

def gradient_descent(cost_function, gradient, initial_guess, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """
    使用梯度下降算法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param gradient: 目标函数的梯度（导数）。
    :param initial_guess: 初始解。
    :param learning_rate: 学习率，控制每次更新的步长。
    :param max_iter: 最大迭代次数。
    :param tolerance: 收敛容忍度。
    :return: 最优解及其目标函数值。
    """
    x = initial_guess
    for i in range(max_iter):
        grad = gradient(x)

        # 更新解
        x_new = x - learning_rate * grad

        # 检查收敛条件
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            return x_new, cost_function(x_new)

        x = x_new

    return x, cost_function(x)


# 示例：使用梯度下降最小化均方误差（MSE）
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


    # 初始猜测值
    initial_guess = np.zeros((X_train.shape[1], 1))

    # 使用梯度下降优化
    best_solution, best_cost = gradient_descent(cost_function, gradient, initial_guess, learning_rate=0.01)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
