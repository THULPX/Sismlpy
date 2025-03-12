import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# 设置matplotlib后端
plt.switch_backend('TkAgg')

# ----------------------------- 深度强化学习（DQN）算法 -----------------------------

# 介绍：
# 深度Q网络（Deep Q-Network, DQN）是一种结合深度学习与Q学习的强化学习算法，主要用于高维状态空间问题。
# DQN使用深度神经网络来近似Q函数，Q函数表示在给定状态下采取某个动作的期望回报。
# 通过与环境的交互，DQN不断优化神经网络，进而提高策略的有效性。
# DQN在多个领域中得到了广泛应用，如游戏AI（例如Atari游戏），机器人控制等。

# 输入输出：
# 输入：
# - 状态：当前环境的状态表示，通常是一个向量或者矩阵（例如图像数据）。
# - 动作：根据当前策略选择的动作，动作空间通常是离散的。
# - 奖励：代理采取某个动作后，环境返回的奖励，用来衡量动作的好坏。
# 输出：
# - Q值：每个动作的Q值，用于评估在当前状态下采取某个动作的期望回报。DQN通过最大化Q值来优化策略。

# 算法步骤：
# 1. 初始化Q网络（当前策略网络）和目标Q网络（用于计算目标Q值）。
# 2. 通过与环境的交互，选择动作并将（状态、动作、奖励、下一个状态、完成标志）存储到经验回放池中。
# 3. 定期从经验回放池中随机采样一个批次，进行Q网络的训练更新。通过Q学习更新Q网络的权重。
# 4. 定期将Q网络的权重复制到目标Q网络，以稳定训练过程。
# 5. 通过最大化Q值来更新策略，确保代理逐步向最优策略靠拢。

# 主要参数：
# - state_size: 状态空间的维度，通常是环境的特征数量。例如，在图像环境中，state_size 可能是图像的像素数。
# - action_size: 动作空间的维度，表示代理可选择的动作数量。例如，在离散动作空间中，action_size表示动作的种类数。
# - learning_rate: 学习率，控制模型权重更新的步长。学习率过大可能导致不稳定，过小则学习效率低。
# - gamma: 折扣因子（discount factor），用于平衡即时奖励和未来奖励的重要性。通常在[0, 1]之间，值越接近1，代理越关注长期回报。
# - epsilon: 探索率，用于平衡探索（随机选择动作）和利用（选择当前Q值最优的动作）之间的权衡。通过epsilon-greedy策略，代理以概率epsilon选择随机动作。
# - max_memory_size: 经验回放池的最大大小，控制存储的经验数量。过大的回放池会增加内存需求，过小则可能导致训练数据不足。
# - batch_size: 批量大小，指每次训练时从经验回放池中随机选择的样本数量。较大的批量可能提高训练效率，但也会增加计算量。
# - epsilon_min: epsilon的最小值，用于确保代理在训练结束时仍有一定的随机探索。
# - epsilon_decay: epsilon衰减因子，用于逐渐减少探索率，增强代理对已学得策略的依赖。

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.state_channels = 2
        self.state_size = state_size
        self.conv_net = nn.Sequential(
            nn.Conv1d(self.state_channels, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self.conv_net(torch.zeros(1, self.state_channels, state_size // self.state_channels)).shape[1]
        self.feature_net = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.advantage_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), self.state_channels, -1)
        conv_features = self.conv_net(x)
        features = self.feature_net(conv_features)
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0

    def __len__(self):
        return len(self.memory)  # 返回内存的实际长度

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.memory[idx] for idx in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.998, memory_size=100000, batch_size=64, soft_update_tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_update_tau = soft_update_tau

        self.memory = PrioritizedReplayBuffer(memory_size)
        self.reward_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.q_history = []

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.HuberLoss(reduction='none')

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.soft_update_tau * param.data + (1.0 - self.soft_update_tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        batch, indices, weights = self.memory.sample(self.batch_size)
        weights = weights.to(self.device)

        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor([float(x[4]) for x in batch]).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        losses = self.criterion(current_q, target_q)
        loss = (losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item(), current_q.mean().item()

    def plot_metrics(self):
        # 在训练完成后一次性绘制图形
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('DQN Training Metrics')

            axes[0, 0].plot(self.reward_history, 'b-', label='Reward')
            axes[0, 0].set_title('Episode Reward')

            axes[0, 1].plot(self.loss_history, 'r-', label='Loss')
            axes[0, 1].set_title('Training Loss')

            axes[1, 0].plot(self.epsilon_history, 'g-', label='Epsilon')
            axes[1, 0].set_title('Exploration Rate')

            axes[1, 1].plot(self.q_history, 'm-', label='Q-Value')
            axes[1, 1].set_title('Average Q-Value')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_metrics: {str(e)}")

class ComplexEnv:
    def __init__(self):
        self.state_size = 8
        self.state = np.zeros(self.state_size)
        self.step_count = 0
        self.max_steps = 200
        self.target_positions = np.array([0.3, -0.2, 0.1, -0.4])

    def reset(self):
        self.state = np.random.uniform(-0.5, 0.5, self.state_size)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        self.step_count += 1
        action_scale = 0.05
        self.state[0:4] += action_scale * (action - 1)
        self.state[4:] = np.tanh(self.state[0:4])

        distances = np.abs(self.state[0:4] - self.target_positions)
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)

        if mean_distance < 0.1:
            reward = 10.0 + (1.0 - min_distance) * 5
        elif mean_distance < 0.3:
            reward = 1.0 - mean_distance + (0.3 - min_distance)
        else:
            reward = -0.1 * mean_distance

        time_penalty = -0.005 * self.step_count
        reward += time_penalty

        done = (self.step_count >= self.max_steps) or (mean_distance < 0.1)
        return self.state.copy(), reward, done


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    env = ComplexEnv()
    state_size = env.state_size
    action_size = 3
    agent = DQN(state_size, action_size)

    episodes = 1000  # 增加训练轮次
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        q_values = []

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss, q_value = agent.replay()
            if loss > 0:
                losses.append(loss)
                q_values.append(q_value)

            if done:
                break

        if episode % 5 == 0:
            agent.update_target_model()

        agent.reward_history.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        agent.loss_history.append(np.mean(losses) if losses else 0)
        agent.q_history.append(np.mean(q_values) if q_values else 0)

        if (episode + 1) % 10 == 0:
            avg_loss = np.mean(losses) if losses else 0
            avg_q = np.mean(q_values) if q_values else 0
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Q-Value: {avg_q:.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")

    # 训练完成后一次性绘制图形
    print("\nFinal Training Results:")
    print(f"Average Reward (last 50 episodes): {np.mean(agent.reward_history[-50:]):.2f}")
    print(f"Average Q-Value (last 50 episodes): {np.mean(agent.q_history[-50:]):.4f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    agent.plot_metrics()