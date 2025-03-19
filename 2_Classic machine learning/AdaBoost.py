import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# ----------------------------- AdaBoost 算法 -----------------------------

# 介绍：
# AdaBoost（Adaptive Boosting）是一种集成学习方法，旨在通过结合多个弱学习器来构建一个强学习器。
# 核心思想:通过反复训练一系列弱学习器，每一轮训练时都着重关注之前学习器未正确分类的数据。
# 在每一轮迭代中，AdaBoost会调整每个样本的权重，重点关注那些被前一轮分类器错误分类的样本。
# 通过不断优化模型的错误，最终实现强大的预测能力。AdaBoost能有效减少过拟合，尤其在小数据集上表现良好。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 样本标签，形状为 (n_samples,)。
# 输出：
# - AdaBoost 模型的预测结果。

# 算法步骤：
# 1. 初始化训练数据集的权重，通常将所有样本的权重设置为相等。
# 2. 在每次迭代中，使用当前权重训练一个弱分类器。
# 3. 根据弱分类器的表现（错误率）调整样本的权重：分类错误的样本权重增加，分类正确的样本权重减少。
# 4. 根据弱分类器的错误率，计算其权重，最终将多个弱分类器组合成一个强分类器。
# 5. 重复步骤2-4，直到达到指定的迭代次数或模型的表现不再提高。
# 6. 使用训练好的AdaBoost模型对新数据进行预测。

# 主要参数：
# - n_estimators: 弱分类器的数量。
# - learning_rate: 每个弱分类器的权重缩放因子。
# - base_estimator: 基学习器，通常是浅层决策树。
# - algorithm: 使用的提升算法（SAMME 或 SAMME.R）。

# 生成多项式数据
np.random.seed(42)
X = np.random.rand(500) * 10
y = np.sin(X) + np.random.normal(0, 0.5, 500)

# 将数据转换成二维数组
X = X[:, np.newaxis]

# 定义基学习器
base_regressor = DecisionTreeRegressor(max_depth=4)

# 定义AdaBoost回归器
regr_1 = AdaBoostRegressor(base_regressor, n_estimators=10, random_state=42)
regr_2 = AdaBoostRegressor(base_regressor, n_estimators=20, random_state=42)
regr_3 = AdaBoostRegressor(base_regressor, n_estimators=100, random_state=42)

# 训练模型
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# 生成预测结果
X_test = np.linspace(0, 10, 500)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Training samples')
plt.plot(X_test, y_1, c='r', label='n_estimators=10', linewidth=2)
plt.plot(X_test, y_2, c='g', label='n_estimators=20', linewidth=2)
plt.plot(X_test, y_3, c='y', label='n_estimators=100', linewidth=2)
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('AdaBoost Regression with Polynomial Features')
plt.legend()
plt.show()
