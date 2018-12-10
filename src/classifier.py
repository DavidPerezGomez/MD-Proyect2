from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from results import Results
import utils
import pickle


class Classifier(ABC):

    def __init__(self, **kwargs):
        self._instances = None
        self._classes = None
        self._class_names = None
        self._model = None
        self._trained_model = None
        self._kwargs = kwargs

    def set_input_format(self, instances, classes):
        self._instances = np.array(instances)
        self._classes = np.array(classes)
        self._class_names = np.unique(classes)

    @abstractmethod
    def train(self, instances, classes, verbose=False):
        pass

    @abstractmethod
    def predict(self, instances):
        pass

    def k_fcv(self, k=10, instances=None, classes=None, save_path_txt=None, save_path_csv=None, verbose=False):
        if verbose:
            print('{}-FOLD CROSS-VALIDATION'.format(k))

        if instances is None or classes is None:
            instances = self._instances
            classes = self._classes
        if instances is not None and classes is not None:
            self.set_input_format(instances, classes)
            instances, classes = utils.parallel_shuffle(np.array(instances), np.array(classes))
            folds = self._make_folds(k=k, instances=instances)
            indiv_results = []
            for i in range(k):
                if verbose:
                    print('Fold {}/{}'.format(i + 1, k))

                test_fold = folds[i]
                train_fold = list(np.setdiff1d(np.array(list(range(len(instances)))),
                                               np.array(test_fold),
                                               assume_unique=True))

                train_instances = instances[train_fold]
                train_classes = classes[train_fold]
                test_instances = instances[test_fold]
                test_classes = classes[test_fold]

                tmp_classifier = self.__class__(**self._kwargs)

                if verbose:
                    print('Entrenando modelo...')

                tmp_classifier.train(train_instances, train_classes, verbose)
                predictions = tmp_classifier.predict(test_instances)

                if verbose:
                    print('Realizando predicción...\n')

                indiv_results.append(Results(predictions, test_classes))

            if save_path_txt:
                self._save_results_txt(indiv_results, save_path_txt)
            if save_path_csv:
                self._save_results_csv(indiv_results, save_path_csv)

            return indiv_results

    def save_model(self, save_path):
        # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence/4529901#4529901
        with open(save_path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def _save_results_txt(self, indiv_results, save_path):
        with open(save_path, 'w') as file:
            file.write(self._results_to_text(indiv_results))

    def _save_results_csv(self, indiv_results, save_path):
        self._results_to_dataframe(indiv_results).to_csv(save_path)

    def _results_to_text(self, indiv_results):
        text_results = {}
        num_results = len(indiv_results)
        class_names = np.unique(self._classes)
        num_classes = len(class_names)
        cum_precision, cum_recall, cum_accuracy, cum_f_score, cum_kappa, \
            cum_tpr, cum_fnr, cum_fpr, cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for class_name in class_names:
            class_cum_precision, class_cum_recall, class_cum_accuracy, class_cum_f_score, class_cum_kappa, \
                class_cum_tpr, class_cum_fnr, class_cum_fpr, class_cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0

            for results in indiv_results:
                class_cum_precision += results.precision(class_name)
                class_cum_recall += results.recall(class_name)
                class_cum_accuracy += results.accuracy(class_name)
                class_cum_f_score += results.f_score(class_name)
                class_cum_kappa += results.kappa(class_name)
                class_cum_tpr += results.tpr(class_name)
                class_cum_fnr += results.fnr(class_name)
                class_cum_fpr += results.fpr(class_name)
                class_cum_tnr += results.tnr(class_name)

            class_avg_precision = class_cum_precision / num_results
            class_avg_recall = class_cum_recall / num_results
            class_avg_accuracy = class_cum_accuracy / num_results
            class_avg_f_score = class_cum_f_score / num_results
            class_avg_kappa = class_cum_kappa / num_results
            class_avg_tpr = class_cum_tpr / num_results
            class_avg_fnr = class_cum_fnr / num_results
            class_avg_fpr = class_cum_fpr / num_results
            class_avg_tnr = class_cum_tnr / num_results

            cum_precision += class_avg_precision
            cum_recall += class_avg_recall
            cum_accuracy += class_avg_accuracy
            cum_f_score += class_avg_f_score
            cum_kappa += class_avg_kappa
            cum_tpr += class_avg_tpr
            cum_fnr += class_avg_fnr
            cum_fpr += class_avg_fpr
            cum_tnr += class_avg_tnr

            tmp_text = '=========================\n'
            tmp_text += 'CLASE: {}\n'.format(class_name.upper())
            tmp_text += '=========================\n'
            tmp_text += 'Valores medios:\n'
            tmp_text += 'Prec.(%): \t{}\n'.format(class_avg_precision)
            tmp_text += 'Recall: \t{}\n'.format(class_avg_recall)
            tmp_text += 'Accuracy: \t{}\n'.format(class_avg_accuracy)
            tmp_text += 'F-Score: \t{}\n'.format(class_avg_f_score)
            tmp_text += 'Kappa: \t\t{}\n'.format(class_avg_kappa)
            tmp_text += 'TPR: \t\t{}\n'.format(class_avg_tpr)
            tmp_text += 'FNR: \t\t{}\n'.format(class_avg_fnr)
            tmp_text += 'FPR: \t\t{}\n'.format(class_avg_fpr)
            tmp_text += 'TNR: \t\t{}\n'.format(class_avg_tnr)

            text_results[class_name] = tmp_text

        avg_precision = cum_precision / num_classes
        avg_recall = cum_recall / num_classes
        avg_accuracy = cum_accuracy / num_classes
        avg_f_score = cum_f_score / num_classes
        avg_kappa = cum_kappa / num_classes
        avg_tpr = cum_tpr / num_classes
        avg_fnr = cum_fnr / num_classes
        avg_fpr = cum_fpr / num_classes
        avg_tnr = cum_tnr / num_classes

        avg_text = '=========================\n'
        avg_text += 'MEDIA DE TODAS LAS CLASES\n'
        avg_text += '=========================\n'
        avg_text += 'Prec.(%): \t{}\n'.format(avg_precision)
        avg_text += 'Recall: \t{}\n'.format(avg_recall)
        avg_text += 'Accuracy: \t{}\n'.format(avg_accuracy)
        avg_text += 'F-Score: \t{}\n'.format(avg_f_score)
        avg_text += 'Kappa: \t\t{}\n'.format(avg_kappa)
        avg_text += 'TPR: \t\t{}\n'.format(avg_tpr)
        avg_text += 'FNR: \t\t{}\n'.format(avg_fnr)
        avg_text += 'FPR: \t\t{}\n'.format(avg_fpr)
        avg_text += 'TNR: \t\t{}\n'.format(avg_tnr)

        final_text = avg_text
        for class_name in text_results:
            final_text += '\n{}'.format(text_results[class_name])

        return final_text

    def _results_to_dataframe(self, indiv_results):
        num_results = len(indiv_results)
        class_names = np.unique(self._classes)
        num_classes = len(class_names)
        cum_precision, cum_recall, cum_accuracy, cum_f_score, cum_kappa, \
            cum_tpr, cum_fnr, cum_fpr, cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for class_name in class_names:
            class_cum_precision, class_cum_recall, class_cum_accuracy, class_cum_f_score, class_cum_kappa, \
                class_cum_tpr, class_cum_fnr, class_cum_fpr, class_cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0

            for results in indiv_results:
                class_cum_precision += results.precision(class_name)
                class_cum_recall += results.recall(class_name)
                class_cum_accuracy += results.accuracy(class_name)
                class_cum_f_score += results.f_score(class_name)
                class_cum_kappa += results.kappa(class_name)
                class_cum_tpr += results.tpr(class_name)
                class_cum_fnr += results.fnr(class_name)
                class_cum_fpr += results.fpr(class_name)
                class_cum_tnr += results.tnr(class_name)

            class_avg_precision = class_cum_precision / num_results
            class_avg_recall = class_cum_recall / num_results
            class_avg_accuracy = class_cum_accuracy / num_results
            class_avg_f_score = class_cum_f_score / num_results
            class_avg_kappa = class_cum_kappa / num_results
            class_avg_tpr = class_cum_tpr / num_results
            class_avg_fnr = class_cum_fnr / num_results
            class_avg_fpr = class_cum_fpr / num_results
            class_avg_tnr = class_cum_tnr / num_results

            cum_precision += class_avg_precision
            cum_recall += class_avg_recall
            cum_accuracy += class_avg_accuracy
            cum_f_score += class_avg_f_score
            cum_kappa += class_avg_kappa
            cum_tpr += class_avg_tpr
            cum_fnr += class_avg_fnr
            cum_fpr += class_avg_fpr
            cum_tnr += class_avg_tnr

        avg_precision = cum_precision / num_classes
        avg_recall = cum_recall / num_classes
        avg_accuracy = cum_accuracy / num_classes
        avg_f_score = cum_f_score / num_classes
        avg_kappa = cum_kappa / num_classes
        avg_tpr = cum_tpr / num_classes
        avg_fnr = cum_fnr / num_classes
        avg_fpr = cum_fpr / num_classes
        avg_tnr = cum_tnr / num_classes
        d = {'precision': [avg_precision],
             'recall': [avg_recall],
             'accuracy': [avg_accuracy],
             'f-score': [avg_f_score],
             'kappa': [avg_kappa],
             'tpr': [avg_tpr],
             'fnr': [avg_fnr],
             'fpr': [avg_fpr],
             'tnr': [avg_tnr]}
        df = pd.DataFrame(data=d)
        return df

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
