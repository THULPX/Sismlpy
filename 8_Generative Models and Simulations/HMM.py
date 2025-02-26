import numpy as np

# ----------------------------- Hidden Markov Models (HMM) -----------------------------

# 介绍：
# 隐马尔可夫模型（Hidden Markov Model, HMM）是一种统计模型，用于描述一个由隐状态序列生成的观测序列。HMM假设系统的状态是隐含的且遵循马尔可夫过程（即当前状态仅依赖于前一个状态），同时每个隐状态生成一个观测值。HMM广泛应用于时间序列分析、语音识别、基因序列分析等领域。
# 该模型主要包括以下部分：
# - 状态转移概率矩阵（A）：描述状态间转移的概率。
# - 观测概率矩阵（B）：描述每个状态下观察到某个观测值的概率。
# - 初始状态概率（π）：描述系统开始时的初始状态分布。

# 输入输出：
# 输入：
# - T: 观测序列长度。
# - N: 隐藏状态的数量。
# - M: 观测符号的数量。
# - A: 状态转移概率矩阵。
# - B: 观测概率矩阵。
# - π: 初始状态概率向量。
# 输出：
# - 观测序列的概率：给定模型参数，计算观测序列的概率。

# 算法步骤：
# 1. 初始化参数，状态转移概率、观测概率以及初始状态概率。
# 2. 根据观察到的序列，使用前向算法（Forward Algorithm）计算给定模型参数下观测序列的概率。
# 3. 通过解码算法（Viterbi）找到最可能的隐状态序列。

# 主要参数：
# - A: 状态转移概率矩阵。
# - B: 观测概率矩阵。
# - π: 初始状态概率向量。
# - observation_seq: 观测序列。

class HMM:
    def __init__(self, A, B, pi):
        """
        初始化隐马尔可夫模型。

        :param A: 状态转移概率矩阵。
        :param B: 观测概率矩阵。
        :param pi: 初始状态概率。
        """
        self.A = A  # 状态转移概率矩阵
        self.B = B  # 观测概率矩阵
        self.pi = pi  # 初始状态概率

    def forward(self, observation_seq):
        """
        前向算法：计算给定观测序列的概率。

        :param observation_seq: 观测序列
        :return: 观测序列的总概率
        """
        T = len(observation_seq)  # 观测序列长度
        N = self.A.shape[0]  # 隐状态数
        alpha = np.zeros((T, N))  # 前向变量alpha

        # 初始化alpha值
        first_observation = observation_seq[0]
        alpha[0, :] = self.pi * self.B[:, first_observation]

        # 递推计算alpha
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, observation_seq[t]]

        # 返回观测序列的总概率
        return np.sum(alpha[T - 1, :])

    def viterbi(self, observation_seq):
        """
        Viterbi算法：给定观测序列，找到最可能的隐状态序列。

        :param observation_seq: 观测序列
        :return: 最可能的隐状态序列
        """
        T = len(observation_seq)  # 观测序列长度
        N = self.A.shape[0]  # 隐状态数
        delta = np.zeros((T, N))  # delta变量
        psi = np.zeros((T, N), dtype=int)  # 最优路径

        # 初始化delta值
        first_observation = observation_seq[0]
        delta[0, :] = self.pi * self.B[:, first_observation]

        # 递推计算delta和路径
        for t in range(1, T):
            for j in range(N):
                delta[t, j] = np.max(delta[t - 1, :] * self.A[:, j]) * self.B[j, observation_seq[t]]
                psi[t, j] = np.argmax(delta[t - 1, :] * self.A[:, j])

        # 反向回溯最优路径
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(delta[T - 1, :])

        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path


# ----------------------------- 示例：使用HMM进行推理 -----------------------------

# 设定模型参数
A = np.array([[0.7, 0.3], [0.4, 0.6]])  # 状态转移矩阵
B = np.array([[0.1, 0.4], [0.6, 0.5]])  # 观测概率矩阵
pi = np.array([0.6, 0.4])  # 初始状态概率

# 观测序列：假设观测符号为0和1
observation_seq = [0, 1, 0, 1, 0]

# 初始化HMM模型
hmm = HMM(A, B, pi)

# 使用前向算法计算观测序列的概率
probability = hmm.forward(observation_seq)
print("观测序列的概率:", probability)

# 使用Viterbi算法解码最可能的隐状态序列
hidden_states = hmm.viterbi(observation_seq)
print("最可能的隐状态序列:", hidden_states)
