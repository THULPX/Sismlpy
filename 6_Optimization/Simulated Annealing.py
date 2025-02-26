import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


# ----------------------------- Simulated Annealing 算法 -----------------------------

# 介绍：
# 模拟退火（Simulated Annealing, SA）是一种全局优化算法，灵感来源于固体物质在加热后冷却过程中的退火现象。在优化问题中，模拟退火通过模拟温度逐渐降低的过程来逐步减少搜索空间，找到全局最优解。
# 它通过接受一些较差的解，避免陷入局部最优解，并在搜索过程中探索更广阔的空间。随着温度逐渐降低，算法收敛到一个最优解。

# 输入输出：
# 输入：
# - cost_function: 需要优化的目标函数。
# - initial_solution: 初始解。
# - temperature: 初始温度。
# - cooling_rate: 降温速率。
# - max_iterations: 最大迭代次数。
# 输出：
# - 最优解及其对应的目标函数值。

# 算法步骤：
# 1. 随机选择一个初始解，并计算其目标函数值。
# 2. 在当前解的邻域内随机选择一个新解，计算新解的目标函数值。
# 3. 如果新解更好，直接接受该解；如果新解较差，则根据一定的概率接受该解（该概率随温度逐渐降低）。
# 4. 持续重复上述过程，并逐渐降低温度。
# 5. 当温度降低到某个阈值或达到最大迭代次数时，返回最优解。

# 主要参数：
# - initial_solution: 初始解。
# - temperature: 初始温度，控制搜索过程的随机性。
# - cooling_rate: 降温速率，决定温度下降的快慢。
# - max_iterations: 最大迭代次数。

def simulated_annealing(cost_function, initial_solution, temperature, cooling_rate, max_iterations):
    """
    使用模拟退火算法寻找最优解。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param initial_solution: 初始解。
    :param temperature: 初始温度，控制搜索过程的随机性。
    :param cooling_rate: 降温速率，决定温度下降的快慢。
    :param max_iterations: 最大迭代次数。
    :return: 最优解和对应的目标函数值。
    """
    current_solution = initial_solution
    current_cost = cost_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost

    # 模拟退火过程
    for iteration in range(max_iterations):
        # 生成邻域解
        neighbor_solution = current_solution + np.random.uniform(-1, 1, size=current_solution.shape)

        # 计算邻域解的目标函数值
        neighbor_cost = cost_function(neighbor_solution)

        # 计算接受邻域解的概率
        if neighbor_cost < current_cost:
            # 如果新解更好，接受新解
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        else:
            # 如果新解更差，根据概率接受
            probability = np.exp((current_cost - neighbor_cost) / temperature)
            if random.random() < probability:
                current_solution = neighbor_solution
                current_cost = neighbor_cost

        # 更新最佳解
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        # 降温
        temperature *= cooling_rate

        # 输出当前温度和迭代次数
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Temperature: {temperature:.4f}, Best Cost: {best_cost:.4f}")

    return best_solution, best_cost


# 示例：使用模拟退火来最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # 目标函数：均方误差
    def cost_function(w):
        # 线性回归模型 y = wx
        predictions = X_train.dot(w)
        return mean_squared_error(y_train, predictions)


    # 初始解：随机初始化
    initial_solution = np.random.rand(1)

    # 模拟退火参数
    temperature = 1000
    cooling_rate = 0.995
    max_iterations = 10000

    # 使用模拟退火优化
    best_solution, best_cost = simulated_annealing(cost_function, initial_solution, temperature, cooling_rate,
                                                   max_iterations)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
