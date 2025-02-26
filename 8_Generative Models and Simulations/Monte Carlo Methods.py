import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


# ----------------------------- Monte Carlo Methods -----------------------------

# 介绍：
# Monte Carlo Methods（蒙特卡洛方法）是一类通过随机采样进行计算和决策的方法。它们广泛应用于强化学习中的策略评估和改进。具体来说，蒙特卡洛方法通常用于通过多个回合的试验，估计某一策略的价值函数或状态-动作价值函数。通过观察实际的奖励轨迹，Monte Carlo方法避免了传统动态规划中的递归计算，从而为复杂的强化学习问题提供了解决方案。蒙特卡洛方法通常用于无模型的环境中，依赖于对环境的经验进行估计。

# 输入输出：
# 输入：
# - env: 环境对象，必须包含reset()和step()方法。
# - gamma: 折扣因子，控制未来奖励的权重。
# - max_episodes: 最大训练回合数。
# 输出：
# - 最优策略。

# 算法步骤：
# 1. 初始化策略和状态值函数。
# 2. 对每个回合：
#    a. 使用当前策略与环境交互，收集状态、动作、奖励等。
#    b. 基于收集的轨迹计算每个状态的回报。
#    c. 更新状态值函数（基于回报）。
# 3. 重复步骤2，直到达到最大回合数或收敛。

# 主要参数：
# - gamma: 折扣因子。
# - max_episodes: 最大训练回合数。

class MonteCarlo:
    def __init__(self, env, gamma=0.99, max_episodes=1000):
        """
        蒙特卡洛算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param gamma: 折扣因子，控制未来奖励的权重。
        :param max_episodes: 最大训练回合数。
        """
        self.env = env
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.state_values = {}  # 状态值函数字典
        self.policy = {}  # 当前策略

    def choose_action(self, state):
        """
        根据当前策略选择动作。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        return self.policy.get(state, np.random.choice(self.env.action_space.n))  # epsilon-greedy选择

    def train(self):
        """
        训练蒙特卡洛方法。
        """
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            trajectory = []
            total_reward = 0

            # 生成一个回合轨迹
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, action, reward))  # 记录每个步骤的状态、动作和奖励
                state = next_state
                total_reward += reward

            # 计算每个状态的回报并更新状态值函数
            returns = 0
            for state, action, reward in reversed(trajectory):
                returns = reward + self.gamma * returns
                if state not in self.state_values:
                    self.state_values[state] = []
                self.state_values[state].append(returns)

            # 更新策略
            for state in self.state_values:
                self.state_values[state] = np.mean(self.state_values[state])  # 平均回报

            # 根据当前的状态值函数更新策略
            for state in self.state_values:
                best_action = np.argmax(self.state_values[state])  # 基于值函数选择最佳动作
                self.policy[state] = best_action

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

    def get_policy(self):
        """
        获取训练后的策略。
        """
        return self.policy


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 训练蒙特卡洛方法
    agent = MonteCarlo(env, gamma=0.99, max_episodes=1000)
    agent.train()

    # 输出最终策略
    final_policy = agent.get_policy()
    print("Final Policy:", final_policy)
