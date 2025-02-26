import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# ----------------------------- Genetic Algorithms (GA) -----------------------------

# 介绍：
# 遗传算法（Genetic Algorithms, GA）是一种模拟自然选择过程的优化算法。它通过模拟生物遗传、变异、交叉和选择等机制来进行搜索，并逐代进化，逐步优化解的质量。GA特别适用于复杂的全局优化问题。
# 它基于“适者生存”原则，即通过选择适应度较高的个体进行繁殖，从而得到更好的后代。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - population_size: 种群大小。
# - n_generations: 进化的代数。
# - mutation_rate: 变异概率。
# - crossover_rate: 交叉概率。
# - n_parents: 每一代选择的父母个体数。
# 输出：
# - 最优解及其对应的目标函数值。

# 算法步骤：
# 1. 初始化种群：随机生成若干个个体，个体代表一个潜在解。
# 2. 计算每个个体的适应度：使用目标函数评估个体的优劣。
# 3. 选择操作：根据适应度选择父母，适应度高的个体有更高的概率被选择。
# 4. 交叉操作：选择父母个体进行交叉，生成下一代的个体。
# 5. 变异操作：随机选择一些个体进行变异，产生新的解。
# 6. 终止条件：达到最大代数或找到最优解。

# 主要参数：
# - population_size: 种群大小。
# - n_generations: 最大代数。
# - mutation_rate: 变异率。
# - crossover_rate: 交叉率。
# - n_parents: 每一代选择的父母个体数。

def genetic_algorithm(cost_function, n_generations, population_size, mutation_rate, crossover_rate, n_parents):
    """
    使用遗传算法进行优化。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param n_generations: 最大代数。
    :param population_size: 种群大小。
    :param mutation_rate: 变异概率。
    :param crossover_rate: 交叉概率。
    :param n_parents: 每代选择的父母个体数。
    :return: 最优解及其目标函数值。
    """
    # 初始化种群（每个个体是一个随机解）
    population = np.random.uniform(-10, 10, (population_size, 1))

    # 记录每一代的最优解
    best_solution = None
    best_cost = np.inf

    for generation in range(n_generations):
        # 计算适应度（目标函数值）
        costs = np.array([cost_function(individual) for individual in population])

        # 更新最优解
        min_cost_index = np.argmin(costs)
        if costs[min_cost_index] < best_cost:
            best_cost = costs[min_cost_index]
            best_solution = population[min_cost_index]

        # 选择父母个体（轮盘赌选择）
        parents = population[np.argsort(costs)][:n_parents]

        # 交叉操作：通过父母个体生成子代
        offspring = []
        for i in range(0, n_parents, 2):
            if random.random() < crossover_rate:
                parent1, parent2 = parents[i], parents[i + 1]
                crossover_point = random.randint(1, 1)
                offspring.append(np.concatenate((parent1[:crossover_point], parent2[crossover_point:])))
                offspring.append(np.concatenate((parent2[:crossover_point], parent1[crossover_point:])))
            else:
                offspring.append(parents[i])
                offspring.append(parents[i + 1])

        # 变异操作
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, len(offspring[i]) - 1)
                offspring[i][mutation_point] = np.random.uniform(-10, 10)

        # 创建新种群
        population = np.array(offspring)

        # 输出当前代数和最优解
        if generation % 100 == 0:
            print(f"Generation {generation}, Best Cost: {best_cost:.4f}")

    return best_solution, best_cost


# 示例：使用遗传算法最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 目标函数：均方误差
    def cost_function(w):
        # 线性回归模型 y = wx
        predictions = X_train.dot(w)
        return mean_squared_error(y_train, predictions)

    # 遗传算法参数
    population_size = 50
    n_generations = 1000
    mutation_rate = 0.1
    crossover_rate = 0.9
    n_parents = 20

    # 使用遗传算法优化
    best_solution, best_cost = genetic_algorithm(cost_function, n_generations, population_size,
                                                 mutation_rate, crossover_rate, n_parents)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
