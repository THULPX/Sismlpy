import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical


# ----------------------------- TRPO (Trust Region Policy Optimization) -----------------------------

# 介绍：
# Trust Region Policy Optimization (TRPO) 是一种强化学习算法，它通过在策略优化过程中施加约束来确保策略更新不会偏离当前策略过多。TRPO通过利用二阶优化（比如自然梯度）来控制更新的步长，确保策略更新的“信任域”不会过大，从而提高策略优化的稳定性。TRPO通过最大化目标函数，同时确保KL散度在某个阈值以内，防止策略更新过大，通常比传统的策略梯度方法更稳定。

# 输入输出：
# 输入：
# - env: 环境对象，必须包含reset()和step()方法。
# - alpha: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - delta: KL散度的限制。
# - max_episodes: 最大训练回合数。
# 输出：
# - 最优策略。

# 算法步骤：
# 1. 初始化策略网络（Actor）和值函数网络（Critic）。
# 2. 对每个回合：
#    a. 使用当前策略与环境交互，获取状态、动作、奖励等。
#    b. 使用奖励信号来估计优势函数。
#    c. 计算当前策略与旧策略之间的KL散度。
#    d. 使用二阶优化（自然梯度）来更新策略。
# 3. 重复步骤2，直到达到最大回合数或收敛。

# 主要参数：
# - alpha: 学习率。
# - gamma: 折扣因子。
# - delta: KL散度的限制。
# - max_episodes: 最大训练回合数。

class ActorCriticNetwork(nn.Module):
    """
    TRPO中使用的策略网络（Actor）和值网络（Critic）的联合网络结构。
    """

    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor_fc = nn.Linear(128, output_dim)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actor_output = torch.softmax(self.actor_fc(x), dim=-1)  # 策略输出
        critic_output = self.critic_fc(x)  # 价值输出
        return actor_output, critic_output


class TRPO:
    def __init__(self, env, alpha=0.01, gamma=0.99, delta=0.01, max_episodes=1000, global_network=None):
        """
        TRPO算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param delta: KL散度的限制。
        :param max_episodes: 最大训练回合数。
        :param global_network: 全局网络。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.max_episodes = max_episodes
        self.global_network = global_network
        self.optimizer = optim.Adam(global_network.parameters(), lr=self.alpha)

    def choose_action(self, state):
        """
        根据当前策略选择动作。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        actor_probs, _ = self.global_network(state_tensor)
        action = torch.multinomial(actor_probs, 1).item()  # 使用多项式分布采样动作
        return action

    def compute_advantage(self, rewards, values, next_value, dones):
        """
        计算优势函数（Advantage Function）。

        :param rewards: 当前回合的奖励。
        :param values: 当前回合的价值。
        :param next_value: 下一状态的价值。
        :param dones: 回合是否结束的标志。
        :return: 计算得到的优势。
        """
        advantages = []
        cumulative_reward = next_value
        for reward, value, done in zip(rewards[::-1], values[::-1], dones[::-1]):
            if done:
                cumulative_reward = 0
            cumulative_reward = reward + self.gamma * cumulative_reward
            advantages.insert(0, cumulative_reward - value)
        return torch.FloatTensor(advantages)

    def train(self):
        """
        训练TRPO算法。
        """
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            rewards = []
            values = []
            log_probs = []
            dones = []
            states = []
            actions = []

            while not done:
                states.append(state)
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # 保存经验
                rewards.append(reward)
                values.append(self.global_network(torch.FloatTensor(state).unsqueeze(0))[1].item())
                log_probs.append(torch.log(self.global_network(torch.FloatTensor(state).unsqueeze(0))[0][action]))
                dones.append(done)
                actions.append(action)
                state = next_state
                total_reward += reward

            # 计算优势
            next_value = self.global_network(torch.FloatTensor(next_state).unsqueeze(0))[1].item()
            advantages = self.compute_advantage(rewards, values, next_value, dones)

            # 更新策略
            self.update_policy(states, actions, rewards, values, advantages, log_probs)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

    def update_policy(self, states, actions, rewards, values, advantages, log_probs):
        """
        使用TRPO的自然梯度来更新策略。
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        advantages = torch.FloatTensor(advantages)
        log_probs = torch.stack(log_probs)

        # 计算策略和价值的损失
        actor_probs, critic_value = self.global_network(states)
        critic_value = critic_value.squeeze()

        # 计算新的log概率
        new_log_probs = torch.log(actor_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # 计算策略损失
        ratio = torch.exp(new_log_probs - log_probs)
        actor_loss = -torch.sum(ratio * advantages)

        # 计算价值损失
        critic_loss = torch.sum((rewards - critic_value) ** 2)

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss

        # 计算KL散度
        old_actor_probs = torch.exp(log_probs)
        kl_divergence = torch.sum(old_actor_probs * (log_probs - new_log_probs))

        # 如果KL散度超过设定的delta，进行步长调整
        if kl_divergence > self.delta:
            return  # 如果KL散度过大，则跳过此次更新

        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 初始化全局网络
    global_network = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)

    # 训练TRPO
    agent = TRPO(env, alpha=0.01, gamma=0.99, delta=0.01, max_episodes=1000, global_network=global_network)
    agent.train()
