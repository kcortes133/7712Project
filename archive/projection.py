import numpy as np
import numba, math
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap as umap

def sphereProjEx():
    sns.set(style='white', rc={'figure.figsize':(10,10)})
    digits = sklearn.datasets.load_digits()
    sphere_mapper = umap.UMAP(output_metric='haversine', random_state=42).fit(digits.data)
    #plt.scatter(sphere_mapper.embedding_.T[0], sphere_mapper.embedding_.T[1], c=digits.target, cmap='Spectral')
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=digits.target, cmap='Spectral')
    plt.show()

def sphereProj(cellTypes, data, names, dataNames):
    sns.set(style='white', rc={'figure.figsize':(10,10)})
    sphere_mapper = umap.UMAP(output_metric='haversine', random_state=42).fit(cellTypes)
    test_mapper = sphere_mapper.transform(data)

    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    cTData = {'Subtype': names, 'X': x, 'Y':y, 'Z':z}
    ctDF = pd.DataFrame(cTData)

    xt = np.sin(test_mapper[:, 0]) * np.cos(test_mapper[:, 1])
    yt = np.sin(test_mapper[:, 0]) * np.sin(test_mapper[:, 1])
    zt = np.cos(test_mapper[:, 0])

    dData = {'Name': dataNames, 'X':xt, 'Y':yt, 'Z':zt}
    dDF = pd.DataFrame(dData)

    for indexP, rowP in dDF.iterrows():
        minD = 100
        closest = 'None'
        pointP = [rowP['X'], rowP['Y'], rowP['Z']]
        for indexC, rowC in ctDF.iterrows():
            pointC = [rowC['X'], rowC['Y'], rowC['Z']]
            oldD = minD
            minD = min(minD, math.dist(pointP, pointC))
            #print(rowP['Name'], rowC['Subtype'], math.dist(pointP, pointC))
            if oldD != minD:
                closest = rowC['Subtype']
        print(closest, minD)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=range(len(names)), cmap='hsv')

    ax.legend(names, scatterpoints=len(names), ncol=len(names))
    ax.scatter(xt, yt, zt, c=[4,1,4,1], cmap='nipy_spectral')
    plt.show()
