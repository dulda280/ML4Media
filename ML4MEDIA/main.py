import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':

    pLength = 'petal length (cm)'
    pWidth = 'petal width (cm)'
    spec = 'species'

    irisData = pd.read_csv('Iris.csv')
    # print(irisData)

    irisData.plot.scatter(x=pLength,
                          y=pWidth,
                          c='Red')
    irisData.plot.scatter(x=spec,
                          y=pLength,
                          c='Red')
    plt.show()
    specData = irisData[spec]
    pLengthData = irisData[pLength]
    pWidthData = irisData[pWidth]
    versiW = []
    versiL = []

    setosaW = []
    setosaL = []

    iterator = 0
    for i in specData:
        if i == 'versicolor':
            versiW.append(pWidthData[iterator])
            versiL.append(pLengthData[iterator])
        if i == 'setosa':
            setosaW.append(pWidthData[iterator])
            setosaL.append(pLengthData[iterator])

        iterator += 1

    print('----------- Setosa -------------')
    print(f'Mean Width: {np.mean(setosaW)}')
    print(f'Mean Length: {np.mean(setosaL)}')
    print('--------------------------------')

    print('----------- Versicolor -------------')
    print(f'Mean Width: {np.mean(versiW)}')
    print(f'Mean Length: {np.mean(versiL)}')
    print('------------------------------------')

    print('Correlation matrix: ', irisData.corr())
    # print(specData)

    ldaData = LinearDiscriminantAnalysis()
    X = irisData[[pLength, pWidth]]
    y = irisData['species']
    print('y', y)
    ldaData.fit(X, y)

    print('TEST', X[[1][:]])
    print(ldaData.predict([[1.5, 5]]))
    crossVal = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    evaluate = cross_val_score(ldaData, X, y, scoring='accuracy', cv=crossVal, n_jobs=1)
    print(f'Linear accuracy mean: {np.mean(evaluate)*100}%')

    quadData = QuadraticDiscriminantAnalysis()
    quadData.fit(X,y)
    evaluateQ = cross_val_score(quadData, X, y, scoring='accuracy', cv=crossVal, n_jobs=1)
    print(f'Quad accuracy mean: {np.mean(evaluateQ)*100}%')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
