from classifier import Classifier


class CombinedNN(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trained_model = None
        self._neurons = kwargs['neurons']

