from sklearn.naive_bayes import GaussianNB as NB
from classifier import Classifier


class NaiveBayes(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = NB
        self._trained_model = None

    def train(self, instances, classes):
        super().train(instances, classes)
        self._trained_model = self._model().fit(self._instances, self._classes)

    def predict(self, instances):
        if instances is None:
            instances = self._instances
        return self._trained_model.predict(instances)
