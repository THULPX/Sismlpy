import numpy as np
import random
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ----------------------------- Q-Learning -----------------------------

# 介绍：
# Q-Learning 是一种无模型的强化学习算法，旨在通过与环境的交互来学习最优策略。Q-Learning算法使用Q值（状态-动作值函数）来衡量在某一状态下采取某一动作的长远回报。该算法通过贝尔曼方程迭代更新Q值，逐步提高决策策略的质量。Q-Learning具有收敛性，适用于离散状态空间和动作空间的环境。

# 输入输出：
# 输入：
# - env: 环境类，包含与环境交互的函数（如reset、step）。
# - alpha: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - epsilon: 探索率，用于控制探索与利用的平衡。
# - max_episodes: 最大训练回合数。
# 输出：
# - 学习到的Q值表和最优策略。

# 算法步骤：
# 1. 初始化Q值表 Q(s, a)，将所有Q值设置为0。
# 2. 对每个回合：
#    a. 从初始状态s开始。
#    b. 在当前状态s下，根据ε-贪婪策略选择一个动作a。
#    c. 执行动作a，观察奖励r和下一个状态s'。
#    d. 更新Q值：Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))。
#    e. 如果状态s'是终止状态，结束回合。
# 3. 重复步骤2，直到收敛或达到最大回合数。

# 主要参数：
# - alpha: 学习率，控制新信息的更新程度。
# - gamma: 折扣因子，控制未来奖励的权重。
# - epsilon: 探索率，控制探索与利用的平衡。
# - max_episodes: 最大训练回合数。

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, max_episodes=1000):
        """
        Q-Learning 算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param epsilon: 探索率。
        :param max_episodes: 最大训练回合数。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episodes = max_episodes

        # 初始化Q值表
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        """
        选择一个动作，基于ε-贪婪策略。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用

    def update_q_value(self, state, action, reward, next_state):
        """
        根据Q-Learning更新规则更新Q值。

        :param state: 当前状态。
        :param action: 当前动作。
        :param reward: 当前动作的奖励。
        :param next_state: 执行动作后的下一个状态。
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def train(self):
        """
        训练Q-Learning算法。

        :return: 最优Q值表。
        """
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

        return self.q_table

    def get_optimal_policy(self):
        """
        获取最优策略。

        :return: 最优策略。
        """
        return np.argmax(self.q_table, axis=1)


# 示例：用Q-Learning解决一个简单的环境（例如迷宫环境）
if __name__ == "__main__":
    # 示例环境，这里使用OpenAI的gym库来创建一个简单的环境（例如FrozenLake）
    import gym

    # 创建一个FrozenLake环境
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # 初始化Q-Learning
    agent = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, max_episodes=1000)

    # 训练
    optimal_q_table = agent.train()

    # 获取最优策略
    optimal_policy = agent.get_optimal_policy()
    print("Optimal Policy: ", optimal_policy)

    # 打印Q值表
    print("Q-Table: ")
    print(optimal_q_table)
