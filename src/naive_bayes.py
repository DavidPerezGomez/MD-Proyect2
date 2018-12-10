from sklearn.naive_bayes import GaussianNB as NB
from classifier import Classifier


class NaiveBayes(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = NB
        self._trained_model = None

    def train(self, instances, classes, verbose=False):
        self.set_input_format(instances, classes)
        self._trained_model = self._model().fit(instances, classes)

    def predict(self, instances=None):
        if instances is None:
            instances = self._instances
        return self._trained_model.predict(instances)
