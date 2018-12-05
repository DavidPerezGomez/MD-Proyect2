from abc import ABC, abstractmethod
import numpy as np
import utils


class Classifier(ABC):

    def __init__(self, **kwargs):
        self._instances = None
        self._classes = None
        self._model = None
        self._trained_model = None
        self._kwargs = kwargs

    @abstractmethod
    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes

    @abstractmethod
    def predict(self, instances):
        pass

    def k_fcv(self, k=10, instances=None, classes=None):
        if instances is None or classes is None:
            instances = self._instances
            classes = self._classes
        if instances is not None and classes is not None:
            instances, classes = utils.parallel_shuffle(instances, classes)
            folds = self._make_folds(k=k, instances=instances)
            for i in range(k):
                test_fold = folds[i]
                train_fold = list(np.setdiff1d(np.array(list(range(len(instances)))),
                                               np.array(test_fold),
                                               assume_unique=True))
                train_instances = instances[train_fold]
                train_classes = classes[train_fold]
                test_instances = instances[test_fold]
                test_classes = classes[test_fold]
                tmp_classifier = self.__class__(**self._kwargs)
                tmp_classifier.train(train_instances, train_classes)

    @staticmethod
    def _make_folds(k, instances):
        length = len(instances)
        folds_length = [int(length / k)] * k
        for i in range(length % k):
            folds_length[i] = folds_length[i] + 1

        last = 0
        folds = []
        for i in range(k):
            # se añaden los indices de la partición de test
            folds.append(list(range(last, last + folds_length[i])))
            last = last + folds_length[i]

        return folds

    @staticmethod
    def eval(predictions, real_classes):
        pass
