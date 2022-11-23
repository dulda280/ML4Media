import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#
# Attribute Information:
# 	 1.	lettr	capital letter	(26 values from A to Z)
# 	 2.	x-box	horizontal position of box	(integer)
# 	 3.	y-box	vertical position of box	(integer)
# 	 4.	width	width of box			(integer)
# 	 5.	high 	height of box			(integer)
# 	 6.	onpix	total # on pixels		(integer)
# 	 7.	x-bar	mean x of on pixels in box	(integer)
# 	 8.	y-bar	mean y of on pixels in box	(integer)
# 	 9.	x2bar	mean x variance			(integer)
# 	10.	y2bar	mean y variance			(integer)
# 	11.	xybar	mean x y correlation		(integer)
# 	12.	x2ybr	mean of x * x * y		(integer)
# 	13.	xy2br	mean of x * y * y		(integer)
# 	14.	x-ege	mean edge count left to right	(integer)
# 	15.	xegvy	correlation of x-ege with y	(integer)
# 	16.	y-ege	mean edge count bottom to top	(integer)
# 	17.	yegvx	correlation of y-ege with x	(integer)



def featureAblation(featureList, labelsY):

    for i in range(0, len(featureList)):
        featureList.pop(i)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            featureList, labelsY, test_size=0.20, random_state=42)

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        evaluate2 = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=crossVal, n_jobs=1)
        print(f'KNN accuracy mean (without feature nr. {i}): {np.mean(evaluate2) * 100}%')



if __name__ == '__main__':
    letterData = pd.read_csv('data/letterrecognitiondatacsv.csv')
    print(letterData)
    # [x-box, y-box, width, height, onpix, x-bar, y-bar, x2bar, y2bar, x2ybr, xy2br, x-edg, xedgvy, y-edg, yedgvx]
    features = []
    letters = []

    for i in range(len(letterData)):
        temp = letterData.iloc[i, 0]
        temp = temp.split(';')
        newTemp = [int(temp[i]) for i in range(1, len(temp))]
        letter = temp[0]

        features.append(newTemp)
        letters.append(letter)

    #print(features)
    print(features[0])
    print(features[0][:])
    print(features[1][:])
    print(features[2][:])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, letters, test_size=0.20, random_state=42)

    ldaData = LinearDiscriminantAnalysis()
    ldaData.fit(X_train, y_train)
    crossVal = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    evaluate = cross_val_score(ldaData, X_test, y_test, scoring='accuracy', cv=crossVal, n_jobs=1)
    print(f'Linear accuracy mean: {np.mean(evaluate) * 100}%')

    quadData = QuadraticDiscriminantAnalysis()
    quadData.fit(X_train, y_train)
    evaluateQ = cross_val_score(quadData, X_test, y_test, scoring='accuracy', cv=crossVal, n_jobs=1)
    print(f'Quad accuracy mean: {np.mean(evaluateQ) * 100}%')

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    evaluate2 = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=crossVal, n_jobs=1)

    npFeats = np.asarray(features)
    print(f'KNN accuracy mean: {np.mean(evaluate2) * 100}%')
    sns.kdeplot(data=npFeats)
    plt.show()

    # print(f'Linear classifier confusion matrix: \n {confusion_matrix(y_test, ldaData.predict(X_test))}')
    # print(f'KNN classifier confusion matrix: \n {confusion_matrix(y_test, knn.predict(X_test))}')
