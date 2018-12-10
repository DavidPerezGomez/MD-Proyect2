import utils
import numpy as np
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import accuracy_score
from classifier import Classifier


class CombinedNN(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = MLP
        self._trained_model = None
        self._n_models = kwargs['n_models']
        self._neurons = kwargs['neurons']

    def train(self, instances, classes, verbose=False):
        self.set_input_format(instances, classes)
        self._trained_model = []
        for i in range(self._n_models):
            if verbose:
                print('Creando modelo {}/{}'.format(i+1, self._n_models))

            sample, sample_classes = utils.bootstrap(instances, classes)
            sub_model = self._model(hidden_layer_sizes=self._neurons,
                                    activation='logistic').fit(sample, sample_classes)
            predictions = sub_model.predict(instances)
            accuracy = accuracy_score(classes, predictions)
            self._trained_model.append({'model': sub_model, 'accuracy': accuracy})

    def predict(self, instances=None):
        if instances is None:
            instances = self._instances
        probabilities = {}
        for model in self._trained_model:
            prediction = dict(zip(model['model'].classes_,
                                  model['model'].predict_proba(instances) * model['accuracy']))
            for class_name in self._class_names:
                try:
                    prob = prediction[class_name]
                except KeyError:
                    prob = 0

                try:
                    if probabilities[class_name] is not None:
                        probabilities[class_name] += prob
                    else:
                        probabilities[class_name] = prob
                except KeyError:
                    probabilities[class_name] = prob


        # TODO fix this shit
        return self._class_names[np.argmax([probabilities[class_name] for class_name in self._class_names])]
