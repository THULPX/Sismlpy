import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------- 深度强化学习（DQN）算法 -----------------------------

# 介绍：
# 深度Q网络（Deep Q-Network, DQN）是一种结合深度学习和Q学习的强化学习算法，旨在解决高维状态空间下的强化学习问题。
# DQN使用神经网络来逼近Q函数，Q函数用来表示在给定状态下采取某个动作的预期回报。通过与环境交互，DQN不断更新网络，优化其决策策略。
# DQN广泛应用于游戏、机器人控制等领域。

# 输入输出：
# 输入：
# - 状态：当前环境的状态。
# - 动作：根据当前策略选择的动作。
# - 奖励：采取动作后从环境中获得的奖励。
# 输出：
# - Q值：每个动作的Q值，用于评估动作的好坏。

# 算法步骤：
# 1. 初始化Q网络和目标Q网络。
# 2. 通过与环境交互，选择动作并存储（状态、动作、奖励、下一个状态）。
# 3. 使用存储的经验回放来训练Q网络。
# 4. 定期将Q网络的权重复制到目标Q网络。
# 5. 通过最大化Q值来更新策略。

# 主要参数：
# - state_size: 状态空间的维度。
# - action_size: 动作空间的维度。
# - learning_rate: 学习率。
# - gamma: 折扣因子，控制未来奖励的权重。
# - epsilon: 探索率，用于平衡探索和利用。
# - max_memory_size: 经验回放的最大大小。
# - batch_size: 批量大小。

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, max_memory_size=2000, batch_size=32):
        """
        初始化深度Q网络（DQN）模型。

        :param state_size: 状态空间的维度。
        :param action_size: 动作空间的维度。
        :param learning_rate: 学习率。
        :param gamma: 折扣因子。
        :param epsilon: 探索率。
        :param epsilon_min: 最小探索率。
        :param epsilon_decay: 探索率衰减因子。
        :param max_memory_size: 经验回放的最大大小。
        :param batch_size: 批量大小。
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        # 经验回放存储
        self.memory = []

        # 创建Q网络和目标Q网络
        self.model = self._build_model()
        self.target_model = self._build_model()

        # 将Q网络的权重复制到目标Q网络
        self.update_target_model()

    def _build_model(self):
        """
        创建Q网络的神经网络模型。

        :return: 一个Keras模型。
        """
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))  # 输出每个动作的Q值
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        将Q网络的权重复制到目标Q网络。
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        存储经验到经验回放池。

        :param state: 当前状态。
        :param action: 当前动作。
        :param reward: 当前奖励。
        :param next_state: 下一个状态。
        :param done: 是否结束。
        """
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        根据epsilon-greedy策略选择动作。

        :param state: 当前状态。
        :return: 选择的动作。
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # 随机选择动作（探索）
        q_values = self.model.predict(np.array([state]))  # 选择Q值最大的动作（利用）
        return np.argmax(q_values[0])

    def replay(self):
        """
        从经验回放池中随机选择批量数据，并进行Q网络的训练。
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # 获取目标Q值
        target_q_values = self.target_model.predict(next_states)
        targets = self.model.predict(states)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]  # 如果结束，目标Q值就是奖励
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(target_q_values[i])  # 否则目标Q值是当前奖励加上未来奖励的折扣值

        # 训练Q网络
        self.model.fit(states, targets, epochs=1, verbose=0)

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        加载训练好的模型权重。

        :param name: 模型保存的路径。
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        保存当前训练的模型权重。

        :param name: 模型保存的路径。
        """
        self.model.save_weights(name)


# 示例：创建DQN模型并训练
if __name__ == "__main__":
    # 假设环境状态维度为4，动作空间为2
    state_size = 4
    action_size = 2

    # 创建DQN模型
    dqn = DQN(state_size, action_size)

    # 假设训练循环
    for e in range(1000):  # 假设训练1000个回合
        state = np.random.randn(state_size)  # 模拟环境的初始状态
        done = False
        total_reward = 0

        while not done:
            action = dqn.act(state)  # 选择动作
            next_state = np.random.randn(state_size)  # 模拟环境的下一个状态
            reward = np.random.randn()  # 模拟环境的奖励
            done = random.choice([True, False])  # 随机决定回合是否结束

            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            dqn.replay()  # 进行经验回放并训练

        # 每100回合保存一次模型
        if e % 100 == 0:
            dqn.save(f"dqn_model_{e}.h5")
            print(f"Episode {e}: Total Reward = {total_reward}")

    # 加载训练好的模型
    dqn.load("dqn_model_900.h5")
    print("Model Loaded")
