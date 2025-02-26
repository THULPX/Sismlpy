import numpy as np
import heapq


# ----------------------------- Dijkstra Algorithm (Shortest Path) -----------------------------

# 介绍：
# Dijkstra算法是一种用于计算图中两个节点之间的最短路径的经典算法。它通过贪心策略，逐步扩展已知的最短路径，直到找到目标节点或所有节点的最短路径。该算法适用于带有非负权重的有向图或无向图。Dijkstra算法通过使用优先队列来高效地找到每个节点的最短路径。

# 输入输出：
# 输入：
# - graph: 图的邻接矩阵或邻接列表表示。
# - start: 起始节点。
# 输出：
# - distances: 从起始节点到每个节点的最短路径距离。
# - previous_nodes: 每个节点的前驱节点（用于路径回溯）。

# 算法步骤：
# 1. 初始化距离数组，设置起始节点的距离为0，其他节点的距离为无限大。
# 2. 使用优先队列（最小堆）存储节点和其当前的最短距离。
# 3. 每次从优先队列中取出当前最短路径的节点，并更新与该节点相邻的节点的最短路径。
# 4. 迭代直到所有节点的最短路径被计算出。

# 主要参数：
# - graph: 图的邻接矩阵或邻接列表。
# - start: 起始节点。

def dijkstra(graph, start):
    """
    计算从起始节点到图中所有其他节点的最短路径。

    :param graph: 图的邻接矩阵或邻接列表表示。
    :param start: 起始节点。
    :return:
        - distances: 从起始节点到每个节点的最短路径距离。
        - previous_nodes: 每个节点的前驱节点，用于路径回溯。
    """
    n = len(graph)
    # 初始化距离数组
    distances = {node: float('inf') for node in range(n)}
    distances[start] = 0
    previous_nodes = {node: None for node in range(n)}  # 记录每个节点的前驱节点

    # 使用优先队列（最小堆）
    pq = [(0, start)]  # (距离, 节点)

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # 如果当前节点的距离已经大于已知最短距离，则跳过
        if current_distance > distances[current_node]:
            continue

        # 遍历与当前节点相邻的节点
        for neighbor, weight in enumerate(graph[current_node]):
            if weight > 0:  # 如果有边
                distance = current_distance + weight

                # 如果找到更短的路径
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

    return distances, previous_nodes


# ----------------------------- 示例：使用Dijkstra算法计算最短路径 -----------------------------

# 创建一个图的邻接矩阵（无向图）
# 0表示没有边，其他数字表示边的权重
graph = np.array([[0, 7, 9, 0, 0, 0],
                  [7, 0, 10, 15, 0, 0],
                  [9, 10, 0, 11, 0, 0],
                  [0, 15, 11, 0, 6, 0],
                  [0, 0, 0, 6, 0, 8],
                  [0, 0, 0, 0, 8, 0]])

# 从节点0开始计算最短路径
start_node = 0
distances, previous_nodes = dijkstra(graph, start_node)

# 输出最短路径
print(f"从节点 {start_node} 到其他节点的最短路径距离：")
for node, distance in distances.items():
    print(f"到节点 {node}: {distance}")


# 输出路径回溯
def get_shortest_path(previous_nodes, start, end):
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.reverse()
    return path


# 获取从节点0到节点5的最短路径
shortest_path = get_shortest_path(previous_nodes, start_node, 5)
print(f"从节点 {start_node} 到节点 5 的最短路径是: {shortest_path}")
