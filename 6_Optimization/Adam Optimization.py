import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


# ----------------------------- Adam Optimization (Adaptive Moment Estimation) -----------------------------

# 介绍：
# Adam（Adaptive Moment Estimation）是一种基于梯度下降的优化算法，它结合了动量法（Momentum）和RMSProp（自适应学习率方法）两者的优点。Adam通过估计一阶矩（梯度的均值）和二阶矩（梯度的方差）来调整每个参数的学习率，从而加速收敛并提高优化效率。Adam非常适合大规模数据集和高维空间的优化问题，尤其在训练深度神经网络时表现优秀。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - gradient: 目标函数的梯度（导数）。
# - initial_guess: 初始解。
# - learning_rate: 学习率。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。
# - beta1: 一阶矩估计的衰减率。
# - beta2: 二阶矩估计的衰减率。
# - epsilon: 防止除零的微小常数。
# 输出：
# - 最优解及其目标函数值。

# 算法步骤：
# 1. 初始化一阶矩（m）和二阶矩（v），它们分别存储梯度的均值和方差。
# 2. 在每次迭代中，计算目标函数的梯度，并更新一阶和二阶矩。
# 3. 更新解：利用Adam公式，调整每个参数的学习率，并更新解。
# 4. 判断收敛：如果梯度的范数小于容忍度，停止迭代。
# 5. 重复步骤 2-4，直到达到最大迭代次数或满足收敛条件。

# 主要参数：
# - cost_function: 目标函数。
# - gradient: 目标函数的梯度。
# - initial_guess: 初始猜测值。
# - learning_rate: 学习率。
# - max_iter: 最大迭代次数。
# - tolerance: 收敛容忍度。
# - beta1: 一阶矩的衰减率。
# - beta2: 二阶矩的衰减率。
# - epsilon: 防止除零的微小常数。

def adam_optimizer(cost_function, gradient, initial_guess, learning_rate=0.001, max_iter=1000, tolerance=1e-6,
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用Adam优化算法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param gradient: 目标函数的梯度（导数）。
    :param initial_guess: 初始解。
    :param learning_rate: 学习率。
    :param max_iter: 最大迭代次数。
    :param tolerance: 收敛容忍度。
    :param beta1: 一阶矩的衰减率。
    :param beta2: 二阶矩的衰减率。
    :param epsilon: 防止除零的微小常数。
    :return: 最优解及其目标函数值。
    """
    # 初始化参数
    x = initial_guess
    m = np.zeros_like(x)  # 一阶矩
    v = np.zeros_like(x)  # 二阶矩
    t = 0  # 迭代次数

    for i in range(max_iter):
        t += 1
        grad = gradient(x)

        # 更新一阶矩和二阶矩
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 更新解
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            return x_new, cost_function(x_new)

        x = x_new

    return x, cost_function(x)


# 示例：使用Adam优化算法最小化均方误差（MSE）
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

    # 使用Adam优化算法优化
    best_solution, best_cost = adam_optimizer(cost_function, gradient, initial_guess, learning_rate=0.001)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
