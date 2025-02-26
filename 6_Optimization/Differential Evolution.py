import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# ----------------------------- Differential Evolution (DE) -----------------------------

# 介绍：
# 差分进化（Differential Evolution, DE）是一种基于群体的随机优化算法。它通过模拟群体在搜索空间中的演化来寻找全局最优解。DE的关键思想是通过当前个体与其他个体之间的差分来生成新的候选解，并用该候选解替换较差的个体。DE适用于连续空间的优化问题，且能够有效避免局部最优解。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - population_size: 种群大小。
# - n_generations: 进化的代数。
# - mutation_factor: 变异因子。
# - crossover_rate: 交叉率。
# 输出：
# - 最优解及其目标函数值。

# 算法步骤：
# 1. 初始化种群：生成若干个随机个体作为初始解。
# 2. 变异操作：从当前种群中随机选择三个个体，计算它们的差分并根据变异因子生成新解。
# 3. 交叉操作：通过交叉操作将变异解与当前解结合，生成新的候选解。
# 4. 选择操作：如果新解的目标函数值优于当前解，则用新解替代当前解。
# 5. 重复上述步骤，直到达到最大代数或找到满意的解。

# 主要参数：
# - population_size: 种群大小。
# - n_generations: 最大代数。
# - mutation_factor: 变异因子，决定变异幅度。
# - crossover_rate: 交叉率。

def differential_evolution(cost_function, n_generations, population_size, mutation_factor, crossover_rate):
    """
    使用差分进化算法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param n_generations: 最大代数。
    :param population_size: 种群大小。
    :param mutation_factor: 变异因子。
    :param crossover_rate: 交叉率。
    :return: 最优解及其目标函数值。
    """
    # 初始化种群
    population = np.random.uniform(-10, 10, (population_size, 1))

    # 记录最优解
    best_solution = None
    best_cost = np.inf

    for generation in range(n_generations):
        # 创建空列表，存储每一代的候选解
        new_population = []

        for i in range(population_size):
            # 选择三个不同的个体进行变异
            candidates = [j for j in range(population_size) if j != i]
            a, b, c = random.sample(candidates, 3)

            # 变异操作
            mutant = population[a] + mutation_factor * (population[b] - population[c])

            # 交叉操作
            crossover = np.random.rand() < crossover_rate
            if crossover:
                trial_solution = mutant
            else:
                trial_solution = population[i]

            # 计算目标函数值
            current_cost = cost_function(trial_solution)

            # 选择操作
            if current_cost < best_cost:
                best_solution = trial_solution
                best_cost = current_cost

            # 将当前解和新解进行比较，更新种群
            if current_cost < cost_function(population[i]):
                new_population.append(trial_solution)
            else:
                new_population.append(population[i])

        # 更新种群
        population = np.array(new_population)

        # 输出当前代数和最优解
        if generation % 100 == 0:
            print(f"Generation {generation}, Best Cost: {best_cost:.4f}")

    return best_solution, best_cost


# 示例：使用差分进化最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 目标函数：均方误差
    def cost_function(solution):
        # 线性回归模型 y = wx
        predictions = X_train.dot(solution)
        return mean_squared_error(y_train, predictions)

    # 差分进化参数
    population_size = 50
    n_generations = 1000
    mutation_factor = 0.8
    crossover_rate = 0.9

    # 使用差分进化优化
    best_solution, best_cost = differential_evolution(cost_function, n_generations, population_size,
                                                       mutation_factor, crossover_rate)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
