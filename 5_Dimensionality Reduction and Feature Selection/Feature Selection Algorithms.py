import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ----------------------------- Feature Selection Algorithms -----------------------------

# 介绍：
# 特征选择（Feature Selection）是一种常用于预处理的技术，旨在选择对模型训练最有用的特征。通过减少不相关的特征，特征选择能够提高模型的训练效率、减少过拟合，并有助于模型解释性。
# 常见的特征选择方法包括：
# 1. **Filter 方法**：通过计算每个特征的统计量来选择特征，如卡方检验（Chi-squared）、方差分析（ANOVA）等。
# 2. **Wrapper 方法**：使用模型性能来评估特征子集的选择，常见的方法有递归特征消除（RFE）。
# 3. **Embedded 方法**：通过算法本身进行特征选择，如Lasso回归、决策树等。

# 输入输出：
# 输入：
# - X: 输入特征，形状为 (n_samples, n_features)。
# - y: 目标标签，形状为 (n_samples,)。
# 输出：
# - 选中的特征，形状为 (n_samples, selected_features)。

# 算法步骤：
# 1. **Filter 方法**：通过计算统计量（例如 F-statistic）评估每个特征的贡献，选择最具信息量的特征。
# 2. **Wrapper 方法**：通过递归地训练模型，评估不同特征子集的性能，选择表现最好的特征组合。
# 3. **Embedded 方法**：利用算法的内置特征选择机制，选择最重要的特征。

# 主要参数：
# - n_features_to_select：选择的特征数量。
# - score_func：选择的评估函数，通常是 ANOVA 或者卡方检验等。

class FeatureSelection:
    def __init__(self, method="filter", n_features_to_select=5, score_func=f_classif):
        """
        初始化特征选择模型。

        :param method: 特征选择方法，"filter"（默认）或 "wrapper"。
        :param n_features_to_select: 选择的特征数量，默认为 5。
        :param score_func: 评估函数，默认为 f_classif（方差分析）。
        """
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.score_func = score_func
        self.selector = None

    def filter_method(self, X_train, y_train):
        """
        使用 Filter 方法（SelectKBest）进行特征选择。
        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标标签。
        :return: 选择后的特征。
        """
        self.selector = SelectKBest(score_func=self.score_func, k=self.n_features_to_select)
        X_new = self.selector.fit_transform(X_train, y_train)
        return X_new

    def wrapper_method(self, X_train, y_train):
        """
        使用 Wrapper 方法（RFE）进行特征选择。
        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标标签。
        :return: 选择后的特征。
        """
        estimator = SVC(kernel="linear")
        self.selector = RFE(estimator, n_features_to_select=self.n_features_to_select)
        X_new = self.selector.fit_transform(X_train, y_train)
        return X_new

    def fit(self, X_train, y_train):
        """
        训练特征选择模型，根据选择的特征选择方法。
        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标标签。
        :return: 选择后的特征。
        """
        if self.method == "filter":
            return self.filter_method(X_train, y_train)
        elif self.method == "wrapper":
            return self.wrapper_method(X_train, y_train)
        else:
            raise ValueError("Method should be either 'filter' or 'wrapper'.")

    def get_selected_features(self):
        """
        获取选择的特征。
        :return: 已选择特征的索引。
        """
        return self.selector.get_support(indices=True)


# 示例：使用鸢尾花数据集进行特征选择和评估
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建特征选择模型，使用 Filter 方法
    feature_selector = FeatureSelection(method="filter", n_features_to_select=2)

    # 训练并选择特征
    X_train_selected = feature_selector.fit(X_train, y_train)
    X_test_selected = X_test[:, feature_selector.get_selected_features()]

    # 使用选择后的特征进行分类（例如 KNN）
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_selected, y_train)
    y_pred = knn.predict(X_test_selected)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
