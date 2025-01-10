

from sklearn.datasets import load_iris, make_blobs
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN模型
from sklearn.naive_bayes import GaussianNB  # 导入朴素贝叶斯模型
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归,最大熵模型
from sklearn.ensemble import AdaBoostClassifier  # 导入 AdaBoost 模型
from sklearn.mixture import GaussianMixture # 导入 EM模型
from hmmlearn.hmm import GaussianHMM  # HMM库
from sklearn.cluster import KMeans  # 导入 K-Means 模型
from sklearn.metrics import precision_score, f1_score, recall_score, silhouette_score, precision_recall_curve, \
    accuracy_score, confusion_matrix, classification_report, average_precision_score  # 导入评估指标
from dlframe import WebManager, Logger
from sklearn import datasets
import math
import matplotlib.pyplot as plt # 绘图
from matplotlib.colors import ListedColormap # 绘图
import numpy as np
from itertools import product  # 确保导入了 itertools 模块


# 鸢尾花数据集
class IrisDataset:
    def __init__(self):
        # 加载鸢尾花数据集
        self.data = datasets.load_iris()  # 加载鸢尾花数据集
        self.x = self.data.data  # 特征矩阵
        self.y = self.data.target  # 标签数组
        self.logger = Logger.get_logger('IrisDataset')
        self.logger.print("Loaded Iris dataset with {} samples".format(len(self.y)))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]  # 返回特征和标签
    def get_labels(self):
        # 提供获取标签的方法
        return self.y

    def get_target_names(self):
        """ 提供获取目标名称的方法 """
        return self.data.target_names.tolist()  # 将 numpy 数组转换为列表

# 合成数据集（聚类数据集）
class SyntheticDataset:
    def __init__(self, n_samples=300, centers=3, cluster_std=1.0, random_state=42):
        # 生成合成数据集
        self.data, self.labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std,
                                            random_state=random_state)
        self.logger = Logger.get_logger('SyntheticDataset')
        self.logger.print("Generated synthetic dataset with {} samples and {} centers".format(n_samples, centers))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]  # 返回特征和标签

# 葡萄酒数据集（聚类数据集）
class WineDataset:
    def __init__(self):
        # 加载葡萄酒数据集
        self.data = datasets.load_wine()  # 加载葡萄酒数据集
        self.x = self.data.data  # 特征矩阵
        self.y = self.data.target  # 标签数组
        self.logger = Logger.get_logger('WineDataset')
        self.logger.print("Loaded Wine dataset with {} samples".format(len(self.y)))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]  # 返回特征和标签

# HMM数据集
class HMMDataset:
    def __init__(self):
        self.sequences = [
            [[0], [1], [0], [2], [3]],
            [[1], [0], [2], [2], [0]],
            [[3], [2], [1], [0], [1]]
        ]
        self.labels = [0, 1, 2]
        self.logger = Logger.get_logger('HMMDataset')
        self.logger.print("Loaded HMM dataset with {} sequences".format(len(self.sequences)))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        # 返回 (sequence, label) 格式
        return self.sequences[idx], self.labels[idx]

# 数据集切分器使用
class TrainTestDataset:
    def __init__(self, item, target_names=None) -> None:
        super().__init__()
        self.item = item
        self.target_names = target_names if target_names is not None else ['setosa', 'versicolor', 'virginica']
    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int):
        return self.item[idx]

    def get_target_names(self):
        return self.target_names
# 数据集切分器
class TestSplitter:
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        # 随机打乱数据集的索引
        indices = np.random.permutation(len(dataset))  # 创建一个随机排列的索引

        # 计算训练集的切分点
        split_index = math.floor(len(dataset) * self.ratio)

        # 基于随机索引划分训练集和测试集
        trainingSet = [dataset[i] for i in indices[:split_index]]  # 训练集
        trainingSet = TrainTestDataset(trainingSet)  # 封装为 TrainTestDataset

        testingSet = [dataset[i] for i in indices[split_index:]]  # 测试集
        testingSet = TrainTestDataset(testingSet)  # 封装为 TrainTestDataset

        self.logger.print("split!")
        # 输出训练集长度
        self.logger.print("training_len = {}".format(len(trainingSet)))
        # 输出训练集内容
        # self.logger.print("training_data = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        return trainingSet, testingSet
# 10个模型
# 1.KNNModel
class KNNModel:
    def __init__(self, n_neighbors=3):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.logger = Logger.get_logger('KNNModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保获取特征和标签时返回的形状一致
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 或根据您的数据维度进行调整

        self.model.fit(features, labels)  # 训练 KNN 模型

    def test(self, testDataset):
        self.logger.print("Testing")

        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions
# 2.朴素贝叶斯Model
class NBModel:
    def __init__(self) -> None:
        super().__init__()
        self.model = GaussianNB()  # 使用高斯朴素贝叶斯分类器
        self.logger = Logger.get_logger('NBModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保从数据集中获取特征和标签
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 训练朴素贝叶斯模型
        self.model.fit(features, labels)

    def test(self, testDataset):
        self.logger.print("Testing")

        # 从数据集中提取特征
        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征

        # 使用朴素贝叶斯模型进行预测
        predictions = self.model.predict(test_features)  # 进行预测
        return predictions  # 返回预测结果
# 3.决策树Model
class DecisionTreeModel:
    def __init__(self, max_depth=None):
        """初始化决策树模型"""
        super().__init__()
        self.model = DecisionTreeClassifier(max_depth=max_depth)  # 使用决策树分类器
        self.logger = Logger.get_logger('DecisionTreeModel')

    def train(self, trainDataset):
        """训练决策树模型"""
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 从数据集中获取特征和标签
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 转换为二维数组

        self.model.fit(features, labels)  # 训练决策树模型

    def test(self, testDataset):
        """在测试数据集上进行预测"""
        self.logger.print("Testing")

        # 从数据集中提取特征
        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)  # 转换为二维数组

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions  # 返回预测结果
# 4.SVMModel 支持向量机
class SVMModel:
    def __init__(self, kernel='linear', C=1.0) -> None:
        """初始化 SVM 模型
        Args:
            kernel: str, optional
                指定支持向量机的内核类型，默认为 'linear'
            C: float, optional
                正则化参数，默认为 1.0
        """
        super().__init__()
        self.model = SVC(kernel=kernel, C=C)  # 使用支持向量机分类器
        self.logger = Logger.get_logger('SVMModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 从数据集中获取特征和标签
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果特征需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 转换为二维数组

        self.model.fit(features, labels)  # 训练 SVM 模型
    def test(self, testDataset):
        self.logger.print("Testing")

        # 从数据集中提取特征
        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)  # 确保是 2D 数组

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions  # 返回预测结果

# 5.逻辑回归Model
class LogisticRegressionModel:
    def __init__(self, C=1.0):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=1000)
        self.logger = Logger.get_logger('LogisticRegressionModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保获取特征和标签时返回的形状一致
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 或根据您的数据维度进行调整

        self.model.fit(features, labels)  # 训练逻辑回归模型

    def test(self, testDataset):
        self.logger.print("Testing")

        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions
# 6.最大熵Model
class MaxEntropyModel:
    def __init__(self, C=1.0):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial')  # 使用最大熵模型
        self.logger = Logger.get_logger('MaxEntropyModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保获取特征和标签时返回的形状一致
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 或根据您的数据维度进行调整

        self.model.fit(features, labels)  # 训练最大熵模型

    def test(self, testDataset):
        self.logger.print("Testing")

        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions
# 7.AdaBoostModel
class AdaBoostModel:
    def __init__(self, n_estimators=50):
        super().__init__()
        self.model = AdaBoostClassifier(n_estimators=n_estimators)
        self.logger = Logger.get_logger('AdaBoostModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保获取特征和标签时返回的形状一致
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征
        labels = np.array([trainDataset[i][1] for i in range(len(trainDataset))])  # 获取标签

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 或根据您的数据维度进行调整

        self.model.fit(features, labels)  # 训练 AdaBoost 模型

    def test(self, testDataset):
        self.logger.print("Testing")

        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions
# 8.EMModel
class EMModel:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components)
        self.logger = Logger.get_logger('EMModel')

    def train(self, trainDataset):
        features = [item[0] for item in trainDataset]
        self.logger.print("Training EM model")
        self.gmm.fit(features)

    def test(self, testDataset):
        features = [item[0] for item in testDataset]
        self.logger.print("Predicting with EM model")
        return self.gmm.predict(features)
# 9.隐马尔科夫Model  ？？
class HiddenMarkovModel:
    def __init__(self, n_components=3):
        self.hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)
        self.logger = Logger.get_logger('HiddenMarkovModel')

    def train(self, trainDataset):
        # trainDataset 的每个元素应为 (features, labels)
        sequences = [item[0] for item in trainDataset]  # 提取特征
        lengths = [len(seq) for seq in sequences]
        flattened_data = [item for seq in sequences for item in seq]
        self.logger.print("Training HMM model")
        self.hmm.fit(flattened_data, lengths)

    def test(self, testDataset):
        sequences = [item[0] for item in testDataset]
        predictions = []
        for seq in sequences:
            predictions.append(self.hmm.predict(seq))
        self.logger.print("Predicting with HMM model")
        return predictions
# 10.KMeansModel    聚类
class KMeansModel:
    def __init__(self, n_clusters=3):
        super().__init__()
        self.model = KMeans(n_clusters=n_clusters)
        self.logger = Logger.get_logger('KMeansModel')

    def train(self, trainDataset):
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 确保获取特征时返回的形状一致
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 从数据集中获取特征

        # 可选：如果 features 需要一个规范的形状
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 或根据您的数据维度进行调整

        self.model.fit(features)  # 训练 K-Means 模型

    def test(self, testDataset):
        self.logger.print("Testing")

        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)

        predictions = self.model.predict(test_features)  # 进行预测
        return predictions


# 结果判别器(分类数据集)  输出准确率、
class TestJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # 提取真实标签
        true_labels = [item[1] for item in test_dataset] # 假设每个 item 是一个 (feature, label) 对
        # 输出预测结果和真实标签
        self.logger.print("(前5个)y_hat = {}...".format([y_hat[i] for i in range(5)]))
        self.logger.print("gt = {}...".format([true_labels[i] for i in range(5)]))

        # 计算并输出准确率
        accuracy = accuracy_score(true_labels, y_hat)
        self.logger.print("Accuracy: {:.2f}%".format(accuracy * 100))

        # 计算查准率 (Precision), 查全率 (Recall), 和 F1 分数
        precision = precision_score(true_labels, y_hat, average='weighted', zero_division=0)
        recall = recall_score(true_labels, y_hat, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, y_hat, average='weighted', zero_division=0)

        self.logger.print("Precision: {:.2f}%".format(precision * 100))
        self.logger.print("Recall: {:.2f}%".format(recall * 100))
        self.logger.print("F1 Score: {:.2f}%".format(f1 * 100))

        # 输出分类报告
        report = classification_report(true_labels, y_hat, target_names=self.get_target_names(), output_dict=True)
        self.logger.print("Classification Report:")
        self.logger.print(report)

        # 输出混淆矩阵
        cm = confusion_matrix(true_labels, y_hat)
        self.logger.print("Confusion Matrix:")
        self.plot_confusion_matrix(cm, target_names=test_dataset.get_target_names())
        # self.logger.print(cm)

        # 绘制 P-R 曲线
        self.plot_precision_recall(true_labels, y_hat)

    def get_target_names(self):
        return ['class 1', 'versicolor', 'virginica']

    def plot_confusion_matrix(self, cm, target_names=None):
        """ 绘制混淆矩阵 """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.get_target_names()))
        plt.xticks(tick_marks, self.get_target_names(), rotation=45)
        plt.yticks(tick_marks, self.get_target_names())

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        else:
            tick_marks = np.arange(cm.shape[0])
            plt.xticks(tick_marks)
            plt.yticks(tick_marks)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def plot_precision_recall(self, true_labels, y_hat):
        """ 绘制 P-R 曲线 """
        n_classes = len(set(true_labels))
        y_test_bin = label_binarize(true_labels, classes=[i for i in range(n_classes)])
        if n_classes == 2:  # 如果是二分类，则需要扩展维度
            y_test_bin = np.hstack((y_test_bin, 1 - y_test_bin))

        # 将预测标签也进行二值化处理
        y_score = label_binarize(y_hat, classes=[i for i in range(n_classes)])
        if n_classes == 2:  # 如果是二分类，则需要扩展维度
            y_score = np.hstack((y_score, 1 - y_score))

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        # 计算微平均 P-R 曲线和面积
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

        # 绘制 P-R 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
                 label='micro-average PR curve (area = {0:0.2f})'.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), ['navy', 'turquoise', 'darkorange']):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label='PR curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    #     # 输出预测结果和真实标签
    #     self.logger.print("(前5个)y_hat = {}...".format([y_hat[i] for i in range(5)]))
    #     self.logger.print("gt = {}...".format([true_labels[i] for i in range(5)]))
    #
    #
    # def plot_precision_recall(self, precision_values, recall_values):
    #     """ 绘制 P-R 曲线 """
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(recall_values, precision_values, marker='o', color='b', label='Precision-Recall Curve')
    #     plt.title('Precision-Recall Curve')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()  # 显示图像
# 结果判别器 (隐马尔科夫数据集)  ？
class HMMJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # gt: 从测试数据集中提取标签
        gt = [item[1] for item in test_dataset]

        # 判断序列预测结果中是否包含目标标签
        correct_predictions = sum(
            1 for i in range(len(y_hat))
            if gt[i] in y_hat[i]  # 检查目标标签是否出现在预测结果中
        )

        # 计算准确率
        accuracy = correct_predictions / len(test_dataset) if len(test_dataset) > 0 else 0

        # 输出日志
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format(gt))
        self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))
# 结果判别器 (聚类数据集) 输出 Silhouette Score
class ClusteringJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('ClusteringJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # 提取真实标签
        true_labels = [item[1] for item in test_dataset]  # 假设每个 item 是一个 (feature, label) 对
        # 输出预测结果和真实标签
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format(true_labels))

        # 计算轮廓系数
        features = np.array([item[0] for item in test_dataset])
        silhouette = silhouette_score(features, y_hat)
        self.logger.print("Silhouette Score = {:.2f}".format(silhouette))

        # 调用绘图函数
        plot_clustering_effect(features, y_hat, title="Clustering Effect Visualization")

def plot_clustering_effect(data, labels, centers=None, title="Clustering Effect"):
    """
    绘制聚类效果

    参数:
    - data: 数据点的坐标，形状为 (n_samples, 2)
    - labels: 每个数据点的标签，长度为 n_samples
    - centers: 聚类中心的坐标，形状为 (n_clusters, 2)，可选
    - title: 图表标题，默认 "Clustering Effect"
    """
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # 创建颜色映射
    cmap = ListedColormap([colors[i] for i in range(len(unique_labels))])

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 黑色用于噪声点
            col = 'k'

        class_member_mask = (labels == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    if centers is not None:
        # 绘制聚类中心
        plt.scatter(centers[:, 0], centers[:, 1], s=250, linewidths=1,
                    marker='X', edgecolors='k', c='white', label='Centers')

    plt.title(title)
    plt.legend()
    plt.show()


#主函数
if __name__ == '__main__':
    with WebManager(parallel=False) as manager:
        #dataset = IrisDataset()
        dataset = manager.register_element('数据集', {

            'Iris':IrisDataset(),
            'Synthetic': SyntheticDataset(n_samples=300, centers=3, cluster_std=1.0, random_state=42),
            'Wine': WineDataset(),
            'HMM': HMMDataset()
        })
        splitter = manager.register_element('数据分割', {
            'ratio:0.8': TestSplitter(0.8),
            'ratio:0.5': TestSplitter(0.5)
        })
        #model = manager.register_element('模型', {'model1': KNNModel(3)})
        model = manager.register_element('模型', {

            'KNN': KNNModel(n_neighbors=5),# 1注册KNN模型
            'NBModel': NBModel(),  # 2注册贝叶斯模型
            'DecisionTree': DecisionTreeModel(max_depth=5),# 3注册决策树模型
            'SVM': SVMModel(kernel='linear', C=1.0),# 4注册支持向量机模型
            'LogisticRegression': LogisticRegressionModel(C=1.0),# 5注册逻辑回归模型
            'MaxEntropy': MaxEntropyModel(C=1.0),  # 6注册最大熵模型
            'AdaBoost': AdaBoostModel(n_estimators=50),  #7 AdaBoost 模型
            'EM': EMModel(),    # 8 EM 模型
            'HiddenMarkovModel': HiddenMarkovModel(n_components=3),# 9隐马尔可夫模型
            'KMeans': KMeansModel(n_clusters=3),  # 10注册 K-Means 模型
        })
        judger = manager.register_element('评价指标', {
            'judger': TestJudger(),
            'clustering_judger': ClusteringJudger(),
            'HMMjudger':HMMJudger()
        })

        train_data_test_data = splitter.split(dataset)
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]
        model.train(train_data)
        y_hat = model.test(test_data)
        judger.judge(y_hat, test_data)
        print(judger)
