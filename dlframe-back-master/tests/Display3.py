import os

from dlframe import WebManager, Logger
from sklearn import datasets
import math
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from hmmlearn import hmm

from hmmlearn.hmm import GaussianHMM  # HMM库



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


# 数据集HMM #
# 数据集 #
'''
class CLUEFineGrainNERDataset:
    def __init__(self):
        # 加载CLUE Fine-Grain NER中文数据集
        # 假设数据集已经解压在当前目录下的'NER'文件夹中
        data_path = 'NER'
        self.data = np.load(data_path, allow_pickle=True)['train']
        self.x = self.data['token']  # 特征矩阵
        self.y = self.data['ner_tags']  # 标签数组
        self.logger = Logger.get_logger('CLUEFineGrainNERDataset')
        self.logger.print("Loaded CLUE Fine-Grain NER dataset with {} samples".format(len(self.y)))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]  # 返回特征和标签

# 人工数据集生成器
def generate_dataset(n_samples=1000):
    states = np.zeros(n_samples, dtype=int)
    states[0] = np.random.choice([0, 1])
    for i in range(1, n_samples):
        if states[i-1] == 0:
            states[i] = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            states[i] = np.random.choice([0, 1], p=[0.4, 0.6])
    observations = np.where(states == 0, np.random.choice([0, 1], size=n_samples), np.random.choice([2, 3], size=n_samples))
    return states, observations
# 数据集 #
class HMMDataset:
    def __init__(self, n_samples=100):
        self.states, self.observations = generate_dataset(n_samples)
        self.logger = Logger.get_logger('HMMDataset')
        self.logger.print("Generated HMM dataset with {} samples".format(len(self.observations)))

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.states[idx]  # 返回观测值和状态


# 隐马尔可夫模型类

class HiddenMarkovModel:
    def __init__(self, n_components=2, n_iter=100):
        self.model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter, init_params='ste')
        self.logger = Logger.get_logger('HiddenMarkovModel')

    def train(self, trainDataset):
        observations = np.array([item[0] for item in trainDataset]).reshape(-1, 1)
        self.model.fit(observations)

    def test(self, testDataset):
        observations = np.array([item[0] for item in testDataset]).reshape(-1, 1)
        return self.model.predict(observations)
# 结果判别器
class TestJudger4:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # 提取真实标签
        true_labels = [item[1] for item in test_dataset]  # 假设每个 item 是一个 (observation, state) 对
        # 输出预测结果和真实标签
        self.logger.print("y_hat = {}".format(y_hat))
        self.logger.print("gt = {}".format(true_labels))

        # 计算准确率
        correct_predictions = sum(1 for i in range(len(y_hat)) if y_hat[i] == true_labels[i])
        accuracy = correct_predictions / len(true_labels) if len(true_labels) > 0 else 0

        # 输出准确率
        self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))
'''
class TestDataset:
    def __init__(self, num) -> None:
        super().__init__()
        self.num = range(num)
        self.logger = Logger.get_logger('TestDataset')
        self.logger.print("I'm in range 0, {}".format(num))

    def __len__(self) -> int:
        return len(self.num)

    def __getitem__(self, idx: int):
        return self.num[idx]
class TrainTestDataset:
    def __init__(self, item) -> None:
        self.item = item

    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int):
        # 直接返回 item，确保格式为二元组
        return self.item[idx]


# 数据集切分器
from sklearn.utils import shuffle
# 数据分割器
class TestSplitter:
    def __init__(self, ratio) -> None:
        self.ratio = ratio
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        dataset = list(dataset)
        shuffled_dataset = shuffle(dataset, random_state=42)
        trainingSet = shuffled_dataset[:math.floor(len(shuffled_dataset) * self.ratio)]
        testingSet = shuffled_dataset[math.floor(len(shuffled_dataset) * self.ratio):]
        self.logger.print("split!")
        self.logger.print("training_len = {}".format(len(trainingSet)))
        return TrainTestDataset(trainingSet), TrainTestDataset(testingSet)

#hmm专用
class HMMSplitter:
    def __init__(self, ratio) -> None:
        self.ratio = ratio
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        # dataset 是一个支持索引访问的对象
        dataset = list(dataset)
        shuffled_dataset = shuffle(dataset, random_state=42)  # 打乱顺序
        split_idx = math.floor(len(shuffled_dataset) * self.ratio)
        trainingSet = shuffled_dataset[:split_idx]
        testingSet = shuffled_dataset[split_idx:]

        # 直接返回数据集的包装类
        self.logger.print("split!")
        self.logger.print("training_len = {}".format(len(trainingSet)))

        return (
            TrainTestDataset(trainingSet),  # 保持原始格式
            TrainTestDataset(testingSet)   # 保持原始格式
        )

    '''
       def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio))]
        trainingSet = TrainTestDataset(trainingSet)

        testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset))]
        testingSet = TrainTestDataset(testingSet)

        self.logger.print("split!")
        self.logger.print("training_len = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        return trainingSet, testingSet
'''


# 模型#
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
# 结果判别器
class TestJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        # self.logger.print("gt = {}".format([item[1] for item in test_dataset]))
        correct_predictions = sum(1 for i in range(len(y_hat)) if y_hat[i] == test_dataset[i])
        accuracy = correct_predictions / len(test_dataset) if len(test_dataset) > 0 else 0
        # 输出准确率
        self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))

class EMJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger2')

    def judge(self, y_hat, test_dataset) -> None:
        # 提取真实标签
        true_labels = [item[1] for item in test_dataset] # 假设每个 item 是一个 (feature, label) 对
        # 输出预测结果和真实标签
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format(true_labels))

        # 计算准确率
        correct_predictions = sum(1 for i in range(len(y_hat)) if y_hat[i] == true_labels[i])
        accuracy = correct_predictions / len(true_labels) if len(true_labels) > 0 else 0

        # 输出准确率
        self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    print(WineDataset())
    with WebManager(parallel=False) as manager:

        dataset = manager.register_element('数据集', {'10_nums': TestDataset(10), 'Iris':IrisDataset(),'Wine': WineDataset(),'HMM':HMMDataset()})
        splitter = manager.register_element('数据分割', {'ratio:0.8': HMMSplitter(0.8), 'ratio:0.5': HMMSplitter(0.5)})
        model = manager.register_element('模型', {'model1': TestModel(1e-3),'EM': EMModel(),'HiddenMarkovModel': HiddenMarkovModel(n_components=3)})
        judger = manager.register_element('评价指标', {'judger1': TestJudger(),'EMjudger': EMJudger(),'HMMjudger':HMMJudger()})

        train_data_test_data = splitter.split(dataset)
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]
        print("Train data sample:", train_data[0])
        print("Test data sample:", test_data[0])
        model.train(train_data)
        y_hat = model.test(test_data)
        judger.judge(y_hat, test_data)


