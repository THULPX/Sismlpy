import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import threading
import time
import torch.nn.functional as F


# ----------------------------- A3C (Asynchronous Advantage Actor-Critic) -----------------------------

# 介绍：
# A3C（Asynchronous Advantage Actor-Critic）是一种强化学习算法，它结合了策略梯度（Actor）和值函数（Critic）的方法，并通过多线程并行训练来加速学习过程。A3C的主要优势是通过多个并行的工作线程来异步更新全局网络，从而提高了样本的多样性，并加速了训练过程。A3C不依赖于经验回放池，而是通过同步每个线程的梯度来更新全局模型。

# 输入输出：
# 输入：
# - env: 环境对象，必须包含reset()和step()方法。
# - alpha: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - max_episodes: 最大训练回合数。
# 输出：
# - 最优策略。

# 算法步骤：
# 1. 初始化全局策略网络（Actor）和全局价值网络（Critic）。
# 2. 对每个线程：
#    a. 使用局部策略网络与环境交互，计算优势和策略梯度。
#    b. 更新全局网络的参数。
# 3. 每个线程异步更新全局模型。
# 4. 更新全局网络后，所有线程使用最新的全局网络进行训练。

# 主要参数：
# - alpha: 学习率。
# - gamma: 折扣因子。
# - max_episodes: 最大训练回合数。

class ActorCriticNetwork(nn.Module):
    """
    结合了策略网络（Actor）和值网络（Critic）的网络结构。
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


class A3C:
    def __init__(self, env, alpha=0.001, gamma=0.99, max_episodes=1000, global_network=None):
        """
        A3C算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param max_episodes: 最大训练回合数。
        :param global_network: 全局网络。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
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
        action = torch.multinomial(actor_probs, 1).item()  # 根据策略分布采样动作
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
        return advantages

    def train(self):
        """
        训练A3C算法。
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

            # 更新全局网络
            self.update_global_network(states, actions, rewards, values, advantages, log_probs)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {total_reward}")

    def update_global_network(self, states, actions, rewards, values, advantages, log_probs):
        """
        更新全局网络。
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

        # 计算策略损失
        action_log_probs = torch.log(actor_probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -torch.sum(action_log_probs * advantages)  # 最小化负的策略梯度

        # 计算价值损失
        critic_loss = torch.sum((rewards - critic_value) ** 2)  # 最小化价值误差

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss

        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# 多线程训练
def worker(env, global_network, alpha, gamma, max_episodes, lock):
    local_agent = A3C(env, alpha, gamma, max_episodes, global_network)
    local_agent.train()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 初始化全局网络
    global_network = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)

    # 多线程并行训练
    num_threads = 4
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker, args=(env, global_network, 0.001, 0.99, 1000, None))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
