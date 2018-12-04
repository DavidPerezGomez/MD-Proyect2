from sklearn.naive_bayes import BaseNB


def naive_bayes(instances, classes):
    classifier = BaseNB()
    prediction = classifier.fit(instances, classes).predict(instances)
    return prediction
