import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# ----------------------------- Particle Swarm Optimization (PSO) 算法 -----------------------------

# 介绍：
# 粒子群优化（Particle Swarm Optimization, PSO）是一种群体智能优化算法，灵感来自鸟群觅食过程。粒子群由多个粒子组成，每个粒子代表一个潜在解。每个粒子根据其当前的位置和速度更新自己的位置，同时也受其历史最优解和整个群体的历史最优解的影响。
# PSO 通过这种方式来搜索解空间，能够在全局范围内找到最优解。

# 输入输出：
# 输入：
# - cost_function: 目标函数，接受一个解并返回其目标值。
# - n_particles: 粒子数量。
# - n_dimensions: 每个粒子的维度。
# - max_iterations: 最大迭代次数。
# - w: 惯性权重，用来平衡粒子的历史速度和当前的搜索方向。
# - c1, c2: 认知和社会学习因子，用于调整粒子根据个人最优解和群体最优解更新位置的权重。
# 输出：
# - 最优解及其对应的目标函数值。

# 算法步骤：
# 1. 初始化粒子群，每个粒子都有位置、速度以及与个人最优解和群体最优解的关联。
# 2. 根据目标函数计算每个粒子的适应度（目标函数值）。
# 3. 更新每个粒子的速度和位置，同时更新每个粒子的个人最优解和全局最优解。
# 4. 重复上述步骤，直到满足终止条件（最大迭代次数或达到最优解）。

# 主要参数：
# - n_particles: 粒子数量。
# - n_dimensions: 每个粒子的维度数。
# - max_iterations: 最大迭代次数。
# - w: 惯性权重，控制粒子的历史速度对当前速度的影响。
# - c1, c2: 认知和社会学习因子，控制粒子根据个人最优解和全局最优解更新位置的权重。

def particle_swarm_optimization(cost_function, n_particles, n_dimensions, max_iterations, w, c1, c2):
    """
    使用粒子群优化算法寻找最优解。

    :param cost_function: 目标函数，接受一个解并返回其目标值。
    :param n_particles: 粒子数量。
    :param n_dimensions: 每个粒子的维度数。
    :param max_iterations: 最大迭代次数。
    :param w: 惯性权重，控制粒子的历史速度对当前速度的影响。
    :param c1: 认知因子，控制粒子根据个人最优解的更新速度。
    :param c2: 社会因子，控制粒子根据全局最优解的更新速度。
    :return: 最优解和对应的目标函数值。
    """
    # 初始化粒子群的位置和速度
    particles_position = np.random.uniform(-10, 10, (n_particles, n_dimensions))
    particles_velocity = np.random.uniform(-1, 1, (n_particles, n_dimensions))

    # 初始化每个粒子的个人最优解和全局最优解
    particles_best_position = particles_position.copy()
    particles_best_cost = np.array([cost_function(p) for p in particles_best_position])
    global_best_position = particles_best_position[np.argmin(particles_best_cost)]
    global_best_cost = np.min(particles_best_cost)

    # 粒子群优化过程
    for iteration in range(max_iterations):
        # 更新每个粒子的速度和位置
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            particles_velocity[i] = (w * particles_velocity[i] +
                                     c1 * r1 * (particles_best_position[i] - particles_position[i]) +
                                     c2 * r2 * (global_best_position - particles_position[i]))

            # 更新位置
            particles_position[i] += particles_velocity[i]

            # 计算新的目标函数值
            current_cost = cost_function(particles_position[i])

            # 更新个人最优解
            if current_cost < particles_best_cost[i]:
                particles_best_cost[i] = current_cost
                particles_best_position[i] = particles_position[i]

        # 更新全局最优解
        current_global_best_cost = np.min(particles_best_cost)
        if current_global_best_cost < global_best_cost:
            global_best_cost = current_global_best_cost
            global_best_position = particles_best_position[np.argmin(particles_best_cost)]

        # 输出当前迭代和全局最优解
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best Cost: {global_best_cost:.4f}")

    return global_best_position, global_best_cost


# 示例：使用粒子群优化最小化均方误差（MSE）
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 目标函数：均方误差
    def cost_function(w):
        # 线性回归模型 y = wx
        predictions = X_train.dot(w)
        return mean_squared_error(y_train, predictions)

    # 粒子群优化参数
    n_particles = 50
    n_dimensions = 1  # 只有一个参数需要优化
    max_iterations = 1000
    w = 0.5  # 惯性权重
    c1 = 1.5  # 认知因子
    c2 = 1.5  # 社会因子

    # 使用粒子群优化
    best_solution, best_cost = particle_swarm_optimization(cost_function, n_particles, n_dimensions,
                                                           max_iterations, w, c1, c2)

    print(f"Optimal Solution: {best_solution}, Best Cost: {best_cost}")

    # 使用最优解进行预测
    final_predictions = X_test.dot(best_solution)
    final_mse = mean_squared_error(y_test, final_predictions)
    print(f"Final MSE on Test Data: {final_mse:.4f}")
