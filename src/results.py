import numpy as np


class Results:

    def __init__(self, predictions, classes):
        self._predictions = predictions
        self._classes = classes
        self._class_names = np.unique(classes)
        self._dict = {'Precision': {},
                      'Accuracy': {},
                      'F-Score': {},
                      'Kappa': {},
                      'TPR': {},
                      'FNR': {},
                      'FPR': {},
                      'TNR': {},
                      'Matrix': {}}
        for key in self._dict:
            subdict = {}
            for class_name in self._class_names:
                subdict[class_name] = None
            self._dict[key] = subdict

    def to_dict(self):
        for class_name in self._class_names:
            self.precision(class_name)
            self.accuracy(class_name)
            self.f_score(class_name)
            self.tpr(class_name)
            self.fnr(class_name)
            self.fpr(class_name)
            self.tnr(class_name)
        return self._dict

    def precision(self, class_name):
        if class_name in self._class_names:
            if self._dict['Precision'][class_name]:
                return self._dict['Precision'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            precision = 100 * cm['tp'] / (cm['tp'] + cm['fp'])
        except ZeroDivisionError:
            precision = 0
        self._dict['Precision'][class_name] = precision
        return precision

    def recall(self, class_name):
        return self.tpr(class_name)

    def accuracy(self, class_name):
        if class_name in self._class_names:
            if self._dict['Accuracy'][class_name]:
                return self._dict['Accuracy'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            accuracy = (cm['tp'] + cm['tn']) / (cm['tp'] + cm['fn'] + cm['fp'] + cm['tn'])
        except ZeroDivisionError:
            accuracy = 0
        self._dict['Accuracy'][class_name] = accuracy
        return accuracy

    def f_measure(self, class_name):
        return self.f_score(class_name)

    def f_score(self, class_name):
        if class_name in self._class_names:
            if self._dict['F-Score'][class_name]:
                return self._dict['F-Score'][class_name]

        try:
            precision = self.precision(class_name)
            recall = self.recall(class_name)
            f_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f_score = 0
        self._dict['F-Score'][class_name] = f_score
        return f_score

    def kappa(self, class_name):
        if class_name in self._class_names:
            if self._dict['Kappa'][class_name]:
                return self._dict['Kappa'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        po = self.accuracy(class_name)

        num_instancias = len(self._predictions)

        # probabilidades a priori de true/false en la predicción
        py_pred = (cm['tp'] + cm['fp']) / num_instancias
        pn_pred = (cm['fn'] + cm['tn']) / num_instancias

        # probabilidades a priori de true/false en la realidad
        py_real = (cm['tp'] + cm['fn']) / num_instancias
        pn_real = (cm['fp'] + cm['tn']) / num_instancias

        # probabilidades de coincidencia de true/false entre la predicción y la realidad
        py = py_pred * py_real
        pn = pn_pred * pn_real

        # probabilidad de coincidencia entre la predicción y la realidad
        pe = py + pn

        try:
            kappa = (po - pe) / (1 - pe)
        except ZeroDivisionError:
            kappa = 0
        self._dict['Kappa'][class_name] = kappa
        return kappa

    def tpr(self, class_name):
        if class_name in self._class_names:
            if self._dict['TPR'][class_name]:
                return self._dict['TPR'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            tpr = cm['tp'] / (cm['tp'] + cm['fn'])
        except ZeroDivisionError:
            tpr = 0
        self._dict['TPR'][class_name] = tpr
        return tpr

    def fnr(self, class_name):
        if class_name in self._class_names:
            if self._dict['FNR'][class_name]:
                return self._dict['FNR'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            fnr = cm['fn'] / (cm['tp'] + cm['fn'])
        except ZeroDivisionError:
            fnr = 0
        self._dict['FNR'][class_name] = fnr
        return fnr

    def fpr(self, class_name):
        if class_name in self._class_names:
            if self._dict['FPR'][class_name]:
                return self._dict['FPR'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            fpr = cm['fp'] / (cm['fp'] + cm['tn'])
        except ZeroDivisionError:
            fpr = 0
        self._dict['FPR'][class_name] = fpr
        return fpr

    def tnr(self, class_name):
        if class_name in self._class_names:
            if self._dict['TNR'][class_name]:
                return self._dict['TNR'][class_name]

            if self._dict['Matrix'][class_name]:
                cm = self._dict['Matrix'][class_name]
            else:
                cm = self._conf_matrix(class_name)
                self._dict['Matrix'][class_name] = cm
        else:
            cm = self._conf_matrix(class_name)

        try:
            tnr = cm['tn'] / (cm['fp'] + cm['tn'])
        except ZeroDivisionError:
            tnr = 0
        self._dict['TNR'][class_name] = tnr
        return tnr

    def _conf_matrix(self, class_name):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(self._predictions)):
            predicted = self._predictions[i]
            real = self._classes[i]
            if predicted == real:
                if predicted == class_name:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted == class_name:
                    fp += 1
                else:
                    fn += 1
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

