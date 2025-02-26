import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


# ----------------------------- Policy Gradient -----------------------------

# 介绍：
# 策略梯度（Policy Gradient）方法是一种强化学习算法，它通过直接优化策略来学习最优的决策策略，而不是通过值函数来间接学习。具体来说，策略梯度方法通过最大化预期累积奖励来调整策略的参数。与Q-Learning等方法不同，策略梯度直接学习一个概率分布，并根据经验调整其参数，从而改善策略的表现。该方法通常用于解决具有连续动作空间的问题。

# 输入输出：
# 输入：
# - env: 环境类，必须包含reset()和step()方法。
# - alpha: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - max_episodes: 最大训练回合数。
# 输出：
# - 最优策略。

# 算法步骤：
# 1. 初始化策略网络。
# 2. 对每个回合：
#    a. 从初始状态s开始。
#    b. 根据当前策略网络生成动作，执行该动作并观察奖励r和下一个状态s'。
#    c. 将状态、动作和奖励存储在回合的经验列表中。
#    d. 计算回合的累积奖励。
#    e. 基于回合的经验更新策略网络。
# 3. 重复步骤2，直到收敛或达到最大回合数。

# 主要参数：
# - alpha: 学习率，控制新信息的更新程度。
# - gamma: 折扣因子，控制未来奖励的权重。
# - max_episodes: 最大训练回合数。

class PolicyNetwork(nn.Module):
    """
    策略网络，采用简单的前馈神经网络作为策略模型。
    """

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  # 返回概率分布


class PolicyGradient:
    def __init__(self, env, alpha=0.001, gamma=0.99, max_episodes=1000):
        """
        策略梯度算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param max_episodes: 最大训练回合数。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.max_episodes = max_episodes

        # 初始化策略网络和优化器
        self.policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.alpha)

    def choose_action(self, state):
        """
        根据当前策略网络选择动作。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probabilities = self.policy_network(state_tensor)
        action = torch.multinomial(probabilities, 1).item()  # 使用多项式分布采样
        return action

    def update_policy(self, rewards, log_probs):
        """
        更新策略网络，基于策略梯度法。

        :param rewards: 该回合的奖励列表。
        :param log_probs: 该回合所有动作的log概率。
        """
        discounted_rewards = []
        cumulative_reward = 0
        for r in rewards[::-1]:
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        log_probs = torch.stack(log_probs)

        # 计算损失
        loss = -torch.sum(log_probs * discounted_rewards)  # 最小化负的策略梯度损失

        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        """
        训练策略梯度算法。
        """
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            rewards = []
            log_probs = []
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # 存储奖励和动作的log概率
                rewards.append(reward)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probabilities = self.policy_network(state_tensor)
                log_prob = torch.log(probabilities[0, action])
                log_probs.append(log_prob)

                state = next_state
                total_reward += reward

            # 更新策略
            self.update_policy(rewards, log_probs)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

        return self.policy_network


# 示例：用策略梯度解决一个简单的环境（例如CartPole）
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 初始化策略梯度
    agent = PolicyGradient(env, alpha=0.001, gamma=0.99, max_episodes=1000)

    # 训练策略梯度
    agent.train()
