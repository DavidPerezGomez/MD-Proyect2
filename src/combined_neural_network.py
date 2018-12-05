from classifier import Classifier


class CombinedNN(Classifier):

    def __init__(self, instances, classes, **kwargs):
        super().__init__(instances, classes, **kwargs)
        self._trained_model = None
        self._neurons = kwargs['neurons']

