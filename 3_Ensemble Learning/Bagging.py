import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------- Bagging（Bootstrap Aggregating）算法 --------------------------

class BaggingModel:
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, n_jobs=1):
        """
        初始化 Bagging 模型。

        :param base_estimator: 基学习器，默认为决策树。
        :param n_estimators: 基学习器的数量，默认为10。
        :param max_samples: 每个基学习器使用的样本数，默认为1.0（表示使用全部样本）。
        :param max_features: 每个基学习器使用的特征数，默认为1.0（表示使用全部特征）。
        :param n_jobs: 并行工作的作业数量，默认为1。
        """
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_train, y_train):
        """
        训练 Bagging 模型。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的目标变量。
        """
        # 初始化并训练 Bagging 模型
        self.model = BaggingClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators,
                                       max_samples=self.max_samples, max_features=self.max_features,
                                       n_jobs=self.n_jobs)

        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, X_test):
        """
        使用训练好的 Bagging 模型进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测结果（分类标签）。
        """
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型的性能。

        :param X_test: 测试数据的特征。
        :param y_test: 测试数据的目标变量（标签）。
        :return: 模型的评估指标，包括准确率、混淆矩阵等。
        """
        predictions = self.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, predictions)

        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test, predictions)

        # 分类报告
        class_report = classification_report(y_test, predictions)

        print(f"Model Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        return accuracy, conf_matrix, class_report


# -------------------- 可视化函数 --------------------

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# 绘制特征重要性
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# 绘制训练过程中的交叉验证准确率
def plot_cv_accuracy(cv_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='--', color='b')
    plt.title('Cross-Validation Accuracy per Fold')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.show()


# -------------------- 示例：使用鸢尾花数据集进行训练和评估 --------------------

if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 Bagging 模型
    bagging_model = BaggingModel(base_estimator=DecisionTreeClassifier(), n_estimators=50, n_jobs=-1)

    # 训练 Bagging 模型
    bagging_model.fit(X_train, y_train)

    # 评估模型
    accuracy, conf_matrix, class_report = bagging_model.evaluate(X_test, y_test)

    # 可视化混淆矩阵
    labels = iris.target_names  # ['setosa', 'versicolor', 'virginica']
    plot_confusion_matrix(conf_matrix, labels)

    # 可视化特征重要性
    plot_feature_importances(bagging_model.model.estimators_[0], iris.feature_names)

    # 计算并绘制交叉验证准确率
    cv_scores = cross_val_score(bagging_model.model, X_train, y_train, cv=5, n_jobs=-1)
    plot_cv_accuracy(cv_scores)
