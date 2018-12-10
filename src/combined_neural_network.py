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
                                    activation='logistic',
                                    max_iter=600).fit(sample, sample_classes)
            predictions = sub_model.predict(instances)
            accuracy = accuracy_score(classes, predictions)
            self._trained_model.append({'model': sub_model, 'accuracy': accuracy})

    def predict(self, instances=None):
        if instances is None:
            instances = self._instances

        predictions = []
        for model in self._trained_model:
            prediction_tmp = model['model'].predict_proba(instances) * model['accuracy']
            prediction = []
            for i in range(len(instances)):
                classes = model['model'].classes_
                pred = prediction_tmp[i]
                prediction.append(dict(zip(classes, pred)))
            predictions.append(prediction)

        probabilities = []
        for i, instance in enumerate(instances):
            probability = {}
            for class_name in self._class_names:
                for prediction in predictions:
                    try:
                        prob = prediction[i][class_name]
                    except KeyError:
                        prob = 0

                    if class_name in probability:
                        probability[class_name] += prob
                    else:
                        probability[class_name] = prob
            probabilities.append(probability)

        results = []
        for probability in probabilities:
            class_name = list(probability.keys())[np.argmax(list(probability.values()))]
            results.append(class_name)

        return np.array(results)
