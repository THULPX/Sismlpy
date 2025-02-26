import numpy as np
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ----------------------------- Deep Q-Network (DQN) -----------------------------

# 介绍：
# 深度Q网络（DQN）是一种结合了深度学习和Q-Learning的强化学习算法。DQN使用神经网络来逼近Q值函数，解决了Q-Learning中对于大规模状态空间和动作空间的限制。它通过深度神经网络来近似状态-动作值函数，从而能够处理高维输入数据（如图像）。DQN通过使用经验回放和目标网络来提高稳定性，避免训练过程中梯度爆炸和过拟合问题。

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
# 1. 初始化Q值网络（深度神经网络）和目标Q值网络（用于稳定训练）。
# 2. 初始化经验回放池（Replay Buffer）。
# 3. 对每个回合：
#    a. 从初始状态s开始。
#    b. 在当前状态s下，根据ε-贪婪策略选择一个动作a。
#    c. 执行动作a，观察奖励r和下一个状态s'。
#    d. 存储(state, action, reward, next_state)到经验回放池。
#    e. 从经验回放池中随机采样一批样本，进行Q值网络的更新。
#    f. 每隔一定回合，将Q值网络的参数复制到目标网络。
# 4. 重复步骤3，直到收敛或达到最大回合数。

# 主要参数：
# - alpha: 学习率，控制新信息的更新程度。
# - gamma: 折扣因子，控制未来奖励的权重。
# - epsilon: 探索率，控制探索与利用的平衡。
# - max_episodes: 最大训练回合数。

class DQN(nn.Module):
    """
    深度Q网络（DQN）模型，用于逼近Q值函数。
    """

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQNetwork:
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=0.1, max_episodes=1000, batch_size=64,
                 replay_buffer_size=10000):
        """
        DQN算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param epsilon: 探索率。
        :param max_episodes: 最大训练回合数。
        :param batch_size: 每次更新时从经验回放池中采样的批次大小。
        :param replay_buffer_size: 经验回放池的最大容量。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size

        # 初始化Q值网络和目标Q值网络
        self.q_network = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_network = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始化目标网络为Q值网络

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # 经验回放池
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

    def choose_action(self, state):
        """
        选择一个动作，基于ε-贪婪策略。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # 探索
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # 利用

    def store_experience(self, state, action, reward, next_state, done):
        """
        存储经验到经验回放池。
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """
        从经验回放池中随机采样一批样本。
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(
            next_states), torch.BoolTensor(dones)

    def update_q_network(self):
        """
        使用经验回放池中的样本更新Q值网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Q值更新公式
        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        # 计算当前状态下的Q值
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        next_q_value = next_q_values.max(1)[0]
        target_q_value = rewards + (self.gamma * next_q_value * (1 - dones))

        # 计算损失
        loss = nn.MSELoss()(q_value, target_q_value)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        每隔一定回合，将Q值网络的参数复制到目标网络。
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        """
        训练DQN算法。
        """
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                self.update_q_network()
                total_reward += reward
                state = next_state

            # 每隔一定回合更新目标网络
            if episode % 10 == 0:
                self.update_target_network()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

        return self.q_network


# 示例：用DQN解决一个简单的环境（例如CartPole）
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 初始化DQN
    agent = DeepQNetwork(env, alpha=0.001, gamma=0.99, epsilon=0.1, max_episodes=1000)

    # 训练DQN
    agent.train()
