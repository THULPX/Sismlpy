import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Markov Chain Monte Carlo (MCMC) -----------------------------

# 介绍：
# Markov Chain Monte Carlo (MCMC) 是一种通过构建马尔可夫链（Markov Chain）来进行随机采样的算法。其主要目标是通过构建一个马尔可夫过程，逐步从复杂分布中生成样本。MCMC通过构建一个链，使得经过足够的时间后，链的分布逼近目标分布。MCMC方法广泛用于统计学、贝叶斯推断和其他机器学习领域，尤其是在需要对复杂概率分布进行采样时。

# 输入输出：
# 输入：
# - target_distribution: 目标分布函数，需要给出密度函数。
# - initial_state: 初始状态。
# - num_samples: 要生成的样本数。
# - burn_in: "烧入期"，前N个样本作为链的初始样本，不用于估计分布。
# - step_size: 每次采样的步长，用于控制提案分布的扩展。

# 输出：
# - samples: 从目标分布中采样得到的样本。

# 算法步骤：
# 1. 初始化状态为初始状态。
# 2. 对于每个采样步骤：
#    a. 根据当前状态从提案分布生成一个候选状态。
#    b. 根据目标分布的概率，决定是否接受候选状态。
#    c. 如果接受候选状态，则更新状态，否则保持当前状态不变。
# 3. 重复步骤2直到生成所需的样本数。

# 主要参数：
# - target_distribution: 目标分布。
# - initial_state: 初始状态。
# - num_samples: 要生成的样本数。
# - burn_in: 烧入期（前N个样本不用于估计分布）。
# - step_size: 提案分布的步长。

class MCMC:
    def __init__(self, target_distribution, initial_state, num_samples=1000, burn_in=200, step_size=0.1):
        """
        Markov Chain Monte Carlo (MCMC) 算法实现。

        :param target_distribution: 目标分布函数（密度函数）。
        :param initial_state: 初始状态。
        :param num_samples: 要生成的样本数。
        :param burn_in: 烧入期，前N个样本作为链的初始样本，不用于估计分布。
        :param step_size: 提案分布的步长。
        """
        self.target_distribution = target_distribution
        self.current_state = initial_state
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.step_size = step_size

    def propose(self):
        """
        从提案分布中生成一个候选状态。
        使用简单的高斯分布来生成新的候选状态。
        """
        return self.current_state + np.random.normal(0, self.step_size)

    def accept(self, proposed_state):
        """
        根据目标分布计算接受概率，并决定是否接受候选状态。
        """
        p_current = self.target_distribution(self.current_state)
        p_proposed = self.target_distribution(proposed_state)
        acceptance_ratio = min(1, p_proposed / p_current)  # 接受比率
        return np.random.rand() < acceptance_ratio

    def run(self):
        """
        运行MCMC算法并采样。
        """
        samples = []
        for _ in range(self.num_samples + self.burn_in):
            proposed_state = self.propose()
            if self.accept(proposed_state):
                self.current_state = proposed_state
            if _ >= self.burn_in:
                samples.append(self.current_state)
        return np.array(samples)

# ----------------------------- 使用MCMC进行采样 -----------------------------

# 示例：使用MCMC从标准正态分布中进行采样

# 目标分布：标准正态分布的概率密度函数
def target_distribution(x):
    return np.exp(-0.5 * x ** 2)

# 初始化MCMC
initial_state = 0.0
mcmc_sampler = MCMC(target_distribution, initial_state, num_samples=10000, burn_in=1000, step_size=0.5)

# 运行MCMC并采样
samples = mcmc_sampler.run()

# 绘制采样结果
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')
x = np.linspace(-5, 5, 1000)
plt.plot(x, np.exp(-0.5 * x ** 2), 'r', lw=2)  # 绘制目标分布
plt.title('MCMC Sampling from Standard Normal Distribution')
plt.show()
