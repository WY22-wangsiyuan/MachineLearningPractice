from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB  # 导入朴素贝叶斯模型
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.mixture import GaussianMixture  # 导入高斯混合模型

#from sklearn.metrics import precision_score, recall_score, f1_score  # 导入评估指标

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
# import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图


from dlframe import WebManager, Logger
from sklearn import datasets
import math
import numpy as np
# import matplotlib.pyplot as plt

# 数据集


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

class TrainTestDataset:
    def __init__(self, features, labels) -> None:
        super().__init__()
        self.features = features  # 特征数据
        self.labels = labels      # 标签数据

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]  # 返回特征和标签

    def get_labels(self):
        # 返回标签
        return self.labels


# 数据集切分器
class TestSplitter:
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: IrisDataset):
        # 随机打乱数据集的索引
        indices = np.random.permutation(len(dataset))  # 创建一个随机排列的索引

        # 计算训练集的切分点
        split_index = math.floor(len(dataset) * self.ratio)

        # 基于随机索引划分训练集和测试集
        training_features = []  # 用于存储训练集特征
        training_labels = []    # 用于存储训练集标签
        for i in indices[:split_index]:
            feature, label = dataset[i]  # 从 dataset 中获取特征和标签
            training_features.append(feature)
            training_labels.append(label)

        trainingSet = TrainTestDataset(training_features, training_labels)  # 封装为 TrainTestDataset

        testing_features = []  # 用于存储测试集特征
        testing_labels = []    # 用于存储测试集标签
        for i in indices[split_index:]:
            feature, label = dataset[i]  # 从 dataset 中获取特征和标签
            testing_features.append(feature)
            testing_labels.append(label)

        testingSet = TrainTestDataset(testing_features, testing_labels)  # 封装为 TrainTestDataset

        self.logger.print("split!")
        self.logger.print("training_set_len = {}".format(len(trainingSet)))
        return trainingSet, testingSet

class TestModel:
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.logger = Logger.get_logger('TestModel')

    def train(self, trainDataset) -> None:
        self.logger.print("trainging, lr = {}, trainDataset = {}".format(self.learning_rate, trainDataset))

    def test(self, testDataset):
        self.logger.print("testing")
        self.logger.imshow(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        return testDataset



class KNNModel:
    def __init__(self, n_neighbors=4):
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


class EMModel:
    def __init__(self, n_components=2, learning_rate=1.0) -> None:
        """初始化高斯混合模型
        Args:
            n_components: int, optional
                混合成分的数量
            learning_rate: float, optional
                学习率（对于 EM 算法通常不需要，但可以保留）
        """
        super().__init__()
        self.model = GaussianMixture(n_components=n_components)  # 使用高斯混合模型
        self.logger = Logger.get_logger('EMModel')

    def train(self, trainDataset) -> None:
        """训练高斯混合模型"""
        self.logger.print("Training with dataset: {}".format(trainDataset))

        # 从数据集中获取特征
        features = np.array([trainDataset[i][0] for i in range(len(trainDataset))])  # 获取特征

        # 训练 GMM 模型
        self.model.fit(features)  # 训练高斯混合模型

    def test(self, testDataset):
        """在测试数据集上进行预测"""
        self.logger.print("Testing")

        # 提取特征进行测试
        test_features = np.array([testDataset[i][0] for i in range(len(testDataset))])  # 获取测试集特征

        # 使用高斯混合模型进行预测（例如预测每个样本属于哪个成分）
        predictions = self.model.predict(test_features)  # 进行预测
        return predictions  # 返回预测结果

# 结果判别器
class TestJudger1:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))


class TestJudger2:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # 提取真实标签
        true_labels = test_dataset.get_labels() # 假设test_dataset是一个包含字典的列表
        # 输出预测结果和真实标签
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format(true_labels))

        # 计算准确率
        correct_predictions = sum(1 for i in range(len(y_hat)) if y_hat[i] == true_labels[i])
        accuracy = correct_predictions / len(true_labels) if len(true_labels) > 0 else 0

        # 输出准确率
        self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))


class TestJudger3:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        """评估模型预测的 Precision、Recall 和 F1 Score
        :param y_hat: 预测值
        :param test_dataset: 真实标签数据集
        """
        # 提取真实标签
        true_labels = test_dataset.get_labels()  # 假设 test_dataset 是一个包含标签的列表

        # 输出预测结果和真实标签
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format(true_labels))

        # 计算查准率 (Precision)
        precision = precision_score(true_labels, y_hat, average='weighted', zero_division=0)

        # 计算查全率 (Recall)
        recall = recall_score(true_labels, y_hat, average='weighted', zero_division=0)

        # 计算 F1 分数
        f1 = f1_score(true_labels, y_hat, average='weighted', zero_division=0)

        # 输出评估结果
        self.logger.print("Precision: {:.2f}%".format(precision * 100))  # 输出查准率
        self.logger.print("Recall: {:.2f}%".format(recall * 100))  # 输出查全率
        self.logger.print("F1 Score: {:.2f}%".format(f1 * 100))  # 输出 F1 分数

        # 计算 P-R 曲线
        precision_values, recall_values, _ = precision_recall_curve(true_labels, y_hat, pos_label=1)  # pos_label 根据需求设定

        # 绘制 P-R 曲线
        # self.plot_precision_recall(precision_values, recall_values)

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




if __name__ == '__main__':
    with WebManager(parallel=False) as manager:

        dataset = manager.register_element('数据集', {'Iris':IrisDataset()})
        splitter = manager.register_element('数据分割', {'ratio:0.8': TestSplitter(0.8), 'ratio:0.5': TestSplitter(0.5)})

        model = manager.register_element('模型', {'model1': TestModel(1e-3),'KNN': KNNModel(n_neighbors=3),'NBModel':NBModel(),'DecisionTree': DecisionTreeModel(max_depth=5),'SVM': SVMModel(kernel='linear', C=1.0),'EM': EMModel(n_components=3)})

        judger = manager.register_element('评价指标', {'列举': TestJudger1(),'准确率': TestJudger2(),'F1':TestJudger3()})

        train_data_test_data = splitter.split(dataset)
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]
        model.train(train_data)
        y_hat = model.test(test_data)
        judger.judge(y_hat, test_data)
        print(judger)
