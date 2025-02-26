import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# ----------------------------- Ant Colony Optimization (ACO) -----------------------------

# 介绍：
# 蚁群优化（Ant Colony Optimization, ACO）是一种模拟蚂蚁寻找食物路径的启发式算法。蚂蚁在寻找路径时，会留下信息素，其他蚂蚁会根据信息素浓度来选择路径。蚁群优化算法利用这一机制进行搜索，并通过信息素的更新和蒸发来引导搜索过程，从而找到最优解。
# ACO通常用于解决组合优化问题，如旅行商问题、调度问题等。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - n_ants: 蚂蚁数量。
# - n_iterations: 迭代次数。
# - alpha: 信息素重要性。
# - beta: 启发函数重要性。
# - evaporation_rate: 信息素蒸发率。
# - initial_pheromone: 初始信息素浓度。
# 输出：
# - 最优解及其目标函数值。

# 算法步骤：
# 1. 初始化蚂蚁的位置和信息素浓度。
# 2. 每只蚂蚁根据信息素和启发函数选择路径，构造解。
# 3. 评估解的质量，更新蚂蚁路径的信息素。
# 4. 信息素蒸发，减少不好的路径的权重。
# 5. 重复上述步骤，直到满足终止条件（最大迭代次数）。

# 主要参数：
# - n_ants: 蚂蚁数量。
# - n_iterations: 迭代次数。
# - alpha: 信息素重要性。
# - beta: 启发函数重要性。
# - evaporation_rate: 信息素蒸发率。
# - initial_pheromone: 初始信息素浓度。

def ant_colony_optimization(cost_function, n_ants, n_iterations, alpha, beta, evaporation_rate, initial_pheromone):
    """
    使用蚁群优化算法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param n_ants: 蚂蚁数量。
    :param n_iterations: 最大迭代次数。
    :param alpha: 信息素重要性。
    :param beta: 启发函数重要性。
    :param evaporation_rate: 信息素蒸发率。
    :param initial_pheromone: 初始信息素浓度。
    :return: 最优解及其目标函数值。
    """
    # 初始化信息素矩阵
    pheromone_matrix = np.full((n_ants, 1), initial_pheromone)

    # 记录最优解
    best_solution = None
    best_cost = np.inf

    for iteration in range(n_iterations):
        all_solutions = []
        all_costs = []

        # 每个蚂蚁构造解
        for ant in range(n_ants):
            solution = []
            # 每个蚂蚁根据信息素选择路径
            while len(solution) < 1:
                probability = pheromone_matrix ** alpha
                # 根据概率选择路径
                next_step = random.choice(range(len(probability)))
                solution.append(next_step)

            # 计算解的目标函数值
            current_cost = cost_function(solution)
            all_solutions.append(solution)
            all_costs.append(current_cost)

            # 更新最优解
            if current_cost < best_cost:
                best_solution = solution
                best_cost = current_cost

        # 更新信息素矩阵
        pheromone_matrix *= (1 - evaporation_rate)  # 信息素蒸发
        for i in range(n_ants):
            for solution in all_solutions:
                pheromone_matrix[solution] += 1.0 / all_costs[i]

        # 输出当前代数和最优解
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best Cost: {best_cost:.4f}")

    return best_solution, best_cost


# 示例：使用蚁群优化最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 目标函数：均方误差
    def cost_function(solution):
        # 线性回归模型 y = wx
        w = solution
        predictions = X_train.dot(w)
        return mean_squared_error(y_train, predictions)

    # 蚁群优化参数
    n_ants = 50
    n_iterations = 1000
    alpha = 1.0  # 信息素重要性
    beta = 2.0   # 启发函数重要性
    evaporation_rate = 0.1
    initial_pheromone = 1.0

    # 使用蚁群优化
    best_solution, best_cost = ant_colony_optimization(cost_function, n_ants, n_iterations, alpha, beta,
                                                        evaporation_rate, initial_pheromone)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
