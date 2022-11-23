import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == '__main__':
    abaloneData = pd.read_csv('abalone.csv')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        #print(abaloneData)
        pass

    length = abaloneData.iloc[:, 0]
    dia = abaloneData.iloc[:, 1]
    height = abaloneData.iloc[:, 2]
    wholeW = abaloneData.iloc[:, 3]
    shuckedW = abaloneData.iloc[:, 4]
    visceraW = abaloneData.iloc[:, 5]
    shelW = abaloneData.iloc[:, 6]
    age = abaloneData.iloc[:, 7]

    y = []
    x = []
    for i in range(0, len(length)):
        x.append([length[i], dia[i], height[i], wholeW[i], shuckedW[i], visceraW[i], shelW[i]])
        y.append(age[i])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.20, random_state=42)



    crossVal = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    #evaluate = cross_val_score(knnReg, X_test, y_test, scoring='neg_root_mean_squared_error', cv=crossVal, n_jobs=1)
    #print(f'Root mean Squared Error: {abs(np.mean(evaluate))}%')

    predictions = []
    kFits = []

    for i in range(1, 100):
        knnReg = KNeighborsRegressor(n_neighbors=i)
        knnReg.fit(X_train, y_train)
        y_predict = knnReg.predict(X_test)
        rootmean = rmse(y_predict, y_test)
        predictions.append(rootmean)
        kFits.append(i)

    plt.plot(kFits, predictions)
    plt.ylabel('RMSE')
    plt.xlabel('Neighbors')
    plt.show()

    for count, x in enumerate(abaloneData.columns):
        plt.subplot(2, 4, count + 1)
        abaloneData[x].plot.density(color='green')
        plt.title(x)

    plt.show()







