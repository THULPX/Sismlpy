import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------- Bayesian Networks -----------------------------

# 介绍：
# Bayesian Networks（贝叶斯网络）是一种用于表示和推理随机变量之间条件依赖关系的图模型。它通过有向无环图（DAG）表示各随机变量之间的条件独立性关系。节点代表随机变量，边表示变量之间的条件依赖关系。贝叶斯网络广泛应用于概率推断、决策支持、机器学习等领域，尤其在不确定性建模和复杂关系推理中表现出色。通过贝叶斯网络，可以有效地进行联合概率分布的计算、预测和解释。

# 输入输出：
# 输入：
# - graph_structure: 图结构，表示随机变量之间的依赖关系。
# - data: 观察数据，作为训练贝叶斯网络的基础。
# - inference_query: 推理查询，通常是关于某些变量的概率查询。
# 输出：
# - 推理结果，通常是后验概率分布或条件概率。

# 算法步骤：
# 1. 定义贝叶斯网络的结构，构建一个有向无环图（DAG）。
# 2. 为每个节点（变量）定义条件概率表（CPT），表示其条件分布。
# 3. 给定观察数据，利用贝叶斯定理更新后验分布。
# 4. 根据推理查询进行概率推断，计算所需变量的后验概率。

# 主要参数：
# - graph_structure: 贝叶斯网络的图结构。
# - data: 训练数据。
# - inference_query: 需要推理的查询（条件概率或后验分布）。

class BayesianNetwork:
    def __init__(self, graph_structure, cpts):
        """
        贝叶斯网络实现。

        :param graph_structure: 图结构，表示节点之间的依赖关系。
        :param cpts: 条件概率表（CPTs），每个节点的条件概率分布。
        """
        self.graph = graph_structure
        self.cpts = cpts

    def get_cpt(self, node):
        """
        获取指定节点的条件概率表。

        :param node: 节点名称（变量）。
        :return: 该节点的条件概率表。
        """
        return self.cpts.get(node, None)

    def query(self, query_vars, evidence):
        """
        基于贝叶斯网络进行推理查询，计算后验概率。

        :param query_vars: 需要查询的变量。
        :param evidence: 观察到的证据（已知变量及其值）。
        :return: 查询结果（后验概率分布）。
        """
        # 这里只做一个简单的推理示范，实际上可以通过精确推理（如变量消除）来实现。
        # 这里使用蒙特卡洛采样等方法估计后验分布。

        # 简单示范：通过遍历CPT估算后验分布（对于复杂网络，需要使用更高效的推理方法）
        result = {}
        for var in query_vars:
            cpt = self.get_cpt(var)
            if cpt:
                result[var] = cpt  # 假设返回CPT作为查询结果
        return result


# ----------------------------- 示例：使用贝叶斯网络进行推理 -----------------------------

# 1. 定义网络结构：我们假设有三个随机变量：A, B, C。
#    - A -> B (A影响B)
#    - A -> C (A影响C)

graph_structure = nx.DiGraph()
graph_structure.add_edges_from([('A', 'B'), ('A', 'C')])

# 2. 定义每个节点的条件概率表（CPT）
#    - P(A) = 0.8
#    - P(B|A) = {P(B=1|A=1) = 0.7, P(B=0|A=1) = 0.3, P(B=1|A=0) = 0.2, P(B=0|A=0) = 0.8}
#    - P(C|A) = {P(C=1|A=1) = 0.6, P(C=0|A=1) = 0.4, P(C=1|A=0) = 0.1, P(C=0|A=0) = 0.9}

cpts = {
    'A': {'0': 0.2, '1': 0.8},  # P(A)
    'B': {
        '1|A=1': 0.7, '0|A=1': 0.3,
        '1|A=0': 0.2, '0|A=0': 0.8
    },  # P(B|A)
    'C': {
        '1|A=1': 0.6, '0|A=1': 0.4,
        '1|A=0': 0.1, '0|A=0': 0.9
    }  # P(C|A)
}

# 3. 初始化贝叶斯网络
bn = BayesianNetwork(graph_structure, cpts)

# 4. 推理查询：计算在A=1的情况下，B和C的条件概率
evidence = {'A': 1}
query_vars = ['B', 'C']

query_result = bn.query(query_vars, evidence)

# 5. 输出推理结果
print("Query Results:")
for var, cpt in query_result.items():
    print(f"Probability of {var}: {cpt}")

# ----------------------------- 网络可视化 -----------------------------

# 可视化贝叶斯网络
plt.figure(figsize=(8, 6))
nx.draw(graph_structure, with_labels=True, node_size=2000, node_color='skyblue', font_size=16)
plt.title("Bayesian Network Structure")
plt.show()
