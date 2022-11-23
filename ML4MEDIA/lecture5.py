import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix



if __name__ == '__main__':
    healtcareData = pd.read_csv('healthcare-dataset-stroke-data.csv')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(healtcareData)
        pass

    gender = 'gender'
    heartDisease = healtcareData.iloc[:, 4]
    everMarried = healtcareData.iloc[:, 5]
    marriedPeople = 0
    neverMarried = 0
    heartDiseaseNneverMarried = 0
    heartDiseaseNMarried = 0
    #print(heartDisease)
    heartDiseaseCounter = 0
    patientsAmount = len(heartDisease)
    for i in range(0, patientsAmount):
        if heartDisease[i] == 1:
            heartDiseaseCounter += 1
        if everMarried[i] == 'Yes':
            marriedPeople += 1
        elif everMarried[i] == 'No':
            neverMarried += 1
        if everMarried[i] == 'No' and heartDisease[i] == 1:
            heartDiseaseNneverMarried += 1
        if everMarried[i] == 'Yes' and heartDisease[i] == 1:
            heartDiseaseNMarried += 1



    print(f'Prior probabillity of having heart disease: {round((heartDiseaseCounter/patientsAmount), 2)*100}%')
    print(f'Prior probabillity of having been married: {round((marriedPeople/patientsAmount), 2)*100}%')
    print(f'Prior probabillity of never being married: {round((neverMarried/patientsAmount), 2)*100}%')
    print(f'Prior probabillity of never being married and having heart disease: {round((heartDiseaseNneverMarried/patientsAmount), 2)*100}%')
    print(f'Prior probabillity of having been married and having heart disease: {round((heartDiseaseNMarried/patientsAmount), 2)*100}%')
    ## Priors can be useful as a baseline indicator of a class if you have an adequate representation of
    # your population in your data
    # However, they are not nessecarily representative of new data
    avgGlucose = healtcareData.iloc[:, 8]
    bmi = healtcareData.iloc[:, 9]
    age = healtcareData.iloc[:, 2]
    stroke = healtcareData.iloc[:, 11]
    hyptertension = healtcareData.iloc[:, 3]
    features = []
    gender = []
    y = []
    for i in range(0, len(bmi)):
        features.append([age[i], avgGlucose[i], heartDisease[i], hyptertension[i]])
        y.append(stroke[i])

    X = features

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.20, random_state=42)


    #y = healtcareData['stroke']
    print('y', y)

    ldaData = LinearDiscriminantAnalysis()
    ldaData.fit(X_train, y_train)
    crossVal = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    evaluate = cross_val_score(ldaData, X_test, y_test, scoring='accuracy', cv=crossVal, n_jobs=1)
    print(f'Linear accuracy mean: {np.mean(evaluate)*100}%')

    quadData = QuadraticDiscriminantAnalysis()
    quadData.fit(X_train, y_train)
    evaluateQ = cross_val_score(quadData, X_test, y_test, scoring='f1', cv=crossVal, n_jobs=1)
    print(f'Quad accuracy mean: {np.mean(evaluateQ) * 100}%')

    #[age, avgGlucose, heartdisease (1/0), hypertension (1/0)]
    patientX = [[65, 95, 0, 1]]


    print(f'Quadratic classifier confusion matrix: \n {confusion_matrix(y_test, quadData.predict(X_test))}')
    print(f'Linear classifier confusion matrix: \n {confusion_matrix(y_test, ldaData.predict(X_test))}')
    print(f'Quadratic classifier prediction: {quadData.predict(patientX)}')
    print(f'Linear classifier prediction: {ldaData.predict(patientX)}')