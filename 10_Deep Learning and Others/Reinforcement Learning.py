import numpy as np
import random

# ----------------------------- Reinforcement Learning -----------------------------

# 介绍：
# 强化学习（Reinforcement Learning, RL）是一类通过与环境交互来学习最优行为策略的机器学习方法。智能体（Agent）在环境中执行动作，并根据环境的反馈（奖励或惩罚）来优化自己的行为策略，目标是最大化累积的奖励。RL的典型问题是马尔可夫决策过程（MDP），常见算法有Q-learning、Policy Gradient等。

# 输入输出：
# 输入：
# - env: 环境。
# - agent: 强化学习智能体。
# 输出：
# - 学习后的策略，智能体在环境中的决策。

# 算法步骤：
# 1. 初始化Q值或策略。
# 2. 在每个时间步骤t，智能体观察当前状态s，并根据策略选择动作a。
# 3. 执行动作a，并观察奖励r和下一个状态s'。
# 4. 更新Q值（在Q-learning中）或策略（在Policy Gradient中）。
# 5. 重复步骤2-4，直到收敛。

# 主要参数：
# - alpha: 学习率。
# - gamma: 折扣因子。
# - epsilon: 探索率。
# - max_episodes: 最大训练回合数。

# ----------------------------- Q-Learning Algorithm -----------------------------

# 介绍：
# Q-learning是一种值迭代方法，使用Q值函数来表示每个状态-动作对的期望回报。Q值函数在不断与环境交互中被更新，最终收敛到最优策略。Q-learning算法具有探索（Exploration）和利用（Exploitation）之间的平衡。

# 输入输出：
# 输入：
# - env: 环境（如Gym环境）。
# - alpha: 学习率。
# - gamma: 折扣因子。
# - epsilon: 探索率。
# - max_episodes: 最大训练回合数。
# 输出：
# - Q: 学习得到的Q值函数。

def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, max_episodes=1000):
    """
    使用Q-learning算法进行强化学习。

    :param env: 环境。
    :param alpha: 学习率。
    :param gamma: 折扣因子。
    :param epsilon: 探索率。
    :param max_episodes: 最大训练回合数。
    :return: 学习得到的Q值函数。
    """
    # 初始化Q值表
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(max_episodes):
        state = env.reset()
        done = False

        while not done:
            # 探索还是利用
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # 探索：随机选择动作
            else:
                action = np.argmax(Q[state])  # 利用：选择Q值最大的动作

            # 执行动作，获得奖励和下一个状态
            next_state, reward, done, _ = env.step(action)

            # 更新Q值
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state

    return Q


# ----------------------------- 示例：使用Q-learning算法解决CartPole问题 -----------------------------

import gym

# 创建CartPole环境
env = gym.make("CartPole-v1")

# 使用Q-learning进行学习
Q_values = q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, max_episodes=1000)

# 输出学习得到的Q值表
print("Q值表：")
print(Q_values)

# 通过训练得到的Q值表来执行最优策略
state = env.reset()
done = False
while not done:
    action = np.argmax(Q_values[state])  # 选择Q值最大的动作
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()

env.close()
