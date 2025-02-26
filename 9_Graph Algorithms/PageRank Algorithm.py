import numpy as np

# ----------------------------- PageRank Algorithm -----------------------------

# 介绍：
# PageRank算法是由Google提出的用于网页排名的一种算法，旨在通过网络结构分析网页的重要性。PageRank的基本思想是通过网页的链接结构，计算每个网页的重要性得分。每个网页的得分由其指向的网页的得分以及指向网页的数量共同决定，反映了一个网页在整个网络中的相对重要性。PageRank广泛用于搜索引擎中排名计算。

# 输入输出：
# 输入：
# - A: 图的邻接矩阵（表示网页之间的链接关系）。
# - d: 阻尼系数，通常取值0.85。
# - max_iter: 最大迭代次数。
# - tol: 收敛阈值。
# 输出：
# - pagerank_scores: 每个节点的PageRank得分。

# 算法步骤：
# 1. 初始化每个网页的PageRank值为1/n，其中n是网页的总数。
# 2. 在每次迭代中，更新每个网页的PageRank值为其所有指向网页PageRank值的加权和。
# 3. 迭代直到每个网页的PageRank值的变化小于给定的收敛阈值或者达到了最大迭代次数。

# 主要参数：
# - A: 邻接矩阵，表示图的结构。
# - d: 阻尼系数，通常取值0.85。
# - max_iter: 最大迭代次数。
# - tol: 收敛阈值。

def pagerank(A, d=0.85, max_iter=100, tol=1e-6):
    """
    使用PageRank算法计算每个节点的得分。

    :param A: 邻接矩阵，表示图的结构。
    :param d: 阻尼系数，默认值为0.85。
    :param max_iter: 最大迭代次数，默认值为100。
    :param tol: 收敛阈值，默认值为1e-6。
    :return: 每个节点的PageRank得分。
    """
    n = A.shape[0]  # 节点数量
    # 计算每个节点的出度
    out_degree = np.sum(A, axis=1)
    out_degree[out_degree == 0] = 1  # 防止除零错误

    # 初始化PageRank值
    pagerank_scores = np.ones(n) / n

    # 转换邻接矩阵为概率矩阵
    M = A / out_degree[:, np.newaxis]

    # 迭代计算PageRank值
    for _ in range(max_iter):
        new_pagerank_scores = (1 - d) / n + d * np.dot(M.T, pagerank_scores)

        # 检查是否收敛
        if np.linalg.norm(new_pagerank_scores - pagerank_scores, 1) < tol:
            break

        pagerank_scores = new_pagerank_scores

    return pagerank_scores


# ----------------------------- 示例：使用PageRank计算节点得分 -----------------------------

# 创建一个简单的邻接矩阵（图结构）
A = np.array([[0, 0, 1, 0],
              [1, 0, 0, 1],
              [0, 1, 0, 1],
              [0, 0, 0, 0]])

# 计算PageRank得分
pagerank_scores = pagerank(A)

# 输出结果
print("节点的PageRank得分：")
for i, score in enumerate(pagerank_scores):
    print(f"节点 {i}: {score}")
