import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


# ----------------------------- PPO (Proximal Policy Optimization) -----------------------------

# 介绍：
# Proximal Policy Optimization (PPO) 是一种强化学习算法，它通过改进的策略优化方法来解决策略更新时出现的训练不稳定问题。PPO基于信赖域优化的思想，在每次更新时，限制策略更新的幅度，从而保证更新的稳定性。通过引入一个剪切的目标函数，PPO避免了过度更新策略的风险，具有较好的性能和收敛性。PPO通常用于离散动作空间和连续动作空间的强化学习任务。

# 输入输出：
# 输入：
# - env: 环境对象，必须包含reset()和step()方法。
# - alpha: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - lamda: GAE(Generalized Advantage Estimation)的lambda值。
# - epsilon: PPO剪切目标函数的epsilon值。
# - max_episodes: 最大训练回合数。
# 输出：
# - 最优策略。

# 算法步骤：
# 1. 初始化策略网络（Actor）和值函数网络（Critic）。
# 2. 对每个回合：
#    a. 使用当前策略与环境交互，获取状态、动作、奖励等。
#    b. 使用GAE（Generalized Advantage Estimation）估计优势。
#    c. 更新策略网络，确保策略更新的幅度在一定范围内（通过剪切目标函数）。
# 3. 重复步骤2，直到达到最大回合数或收敛。

# 主要参数：
# - alpha: 学习率。
# - gamma: 折扣因子。
# - lamda: GAE的lambda值。
# - epsilon: PPO剪切目标函数的epsilon值。
# - max_episodes: 最大训练回合数。

class ActorCriticNetwork(nn.Module):
    """
    PPO中使用的策略网络（Actor）和值网络（Critic）的联合网络结构。
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


class PPO:
    def __init__(self, env, alpha=0.001, gamma=0.99, lamda=0.95, epsilon=0.2, max_episodes=1000, global_network=None):
        """
        PPO算法实现。

        :param env: 环境对象，必须包含reset()和step()方法。
        :param alpha: 学习率。
        :param gamma: 折扣因子。
        :param lamda: GAE的lambda值。
        :param epsilon: PPO剪切目标函数的epsilon值。
        :param max_episodes: 最大训练回合数。
        :param global_network: 全局网络。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
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
        计算优势函数（Advantage Function）使用GAE（Generalized Advantage Estimation）。

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

        advantages = torch.FloatTensor(advantages)
        return advantages

    def train(self):
        """
        训练PPO算法。
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
        使用PPO的剪切目标函数更新策略。
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

        # 计算策略损失（包括剪切目标）
        ratio = torch.exp(new_log_probs - log_probs)
        clip_advantage = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(ratio * advantages, clip_advantage).mean()

        # 计算价值损失
        critic_loss = torch.sum((rewards - critic_value) ** 2)  # 最小化价值误差

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss

        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # 初始化全局网络
    global_network = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)

    # 训练PPO
    agent = PPO(env, alpha=0.001, gamma=0.99, lamda=0.95, epsilon=0.2, max_episodes=1000, global_network=global_network)
    agent.train()
