import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid

if __name__ == '__main__':
    x, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.20, random_state=42)

    print(x[0])

    xsplit, ysplit = np.hsplit(x, indices_or_sections=2)
    print(xsplit[0])
    print(ysplit[0])

    plt.scatter(x=xsplit, y=ysplit)
    plt.show()

    plt.hist(x)
    plt.show()

    nCentroidCLF = NearestCentroid()
    nCentroidCLF.fit(X_train, y_train)

    print(nCentroidCLF.predict([[-3, 3]]))
    # evaluate = cross_val_score(nCentroidCLF, train, scoring='accuracy', cv=crossVal, n_jobs=1)
    # print(evaluate)
    prediction = []
    for i in range(len(y_test)):
        prediction.append(nCentroidCLF.predict(X_test[[i]]))

    print(len(prediction))
    print(len(y_test))

    print(f'Confusion matrix: {sklearn.metrics.confusion_matrix(y_test, prediction)}')
    print(f'Accuracy:{sklearn.metrics.accuracy_score(y_test, prediction) * 100}%')
    print(f'Precision: {sklearn.metrics.average_precision_score(y_test, prediction) * 100}%')
    print(f'Recall: {sklearn.metrics.recall_score(y_test, prediction) * 100}%')
