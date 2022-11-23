import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid

def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results

if __name__ == '__main__':
    longleyData = pd.read_csv('longley.csv')

    print(longleyData)
    employ = 'Employed'
    gnp = 'GNP'

    longleyData.plot.scatter(x=employ,
                             y=gnp,
                             c='Red')
    plt.show()

    employData = longleyData[employ]
    gnpdata = longleyData[gnp]

    pz1 = np.poly1d(np.polyfit(employData, gnpdata, 1))
    pz2 = np.poly1d(np.polyfit(employData, gnpdata, 2))
    pz3 = np.poly1d(np.polyfit(employData, gnpdata, 3))
    pz4 = np.poly1d(np.polyfit(employData, gnpdata, 4))
    pz5 = np.poly1d(np.polyfit(employData, gnpdata, 5))
    pz10 = np.poly1d(np.polyfit(employData, gnpdata, 10))

    xp = np.linspace(min(employData), max(employData))
    plt.xlabel('Employed')
    plt.ylabel('GNP')
    _ = plt.plot(employData, gnpdata, '.', xp, pz4(xp), '-', xp)
    plt.xlim(min(employData) - 5, max(employData) + 5)
    plt.ylim(min(gnpdata) - 10, max(gnpdata) + 10)
    plt.show()


    plt.plot(xp, pz1(xp), color='green')
    plt.plot(xp, pz2(xp), color='blue')
    plt.plot(xp, pz3(xp), color='red')
    plt.plot(xp, pz4(xp), color='yellow')
    plt.plot(xp, pz5(xp), color='purple')
    plt.plot(xp, pz10(xp), color='orange')
    plt.show()

    print(adjR(employData, gnpdata, 1))
    print(adjR(employData, gnpdata, 2))
    print(adjR(employData, gnpdata, 3))
    print(adjR(employData, gnpdata, 4))
    print(adjR(employData, gnpdata, 5))
    print(adjR(employData, gnpdata, 10))

    norm_pz1 = np.linalg.norm(pz1)
    print(norm_pz1)

    newY = []
    newX = []
    for i in range(1, 10):
        newX.append(i)
        r = np.linalg.norm(employData, ord=i)
        print(r)
        newY.append(r)

    plt.scatter(newX, newY, c='red')
    plt.show()




