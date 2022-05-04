import numpy as np
import numba
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap



def sphereMapper(data, colors):
    sns.set(style='white', rc={'figure.figsize':(10,10)})
    sphere_mapper = umap.UMAP(output_metric='haversine', random_state=42).fit(data)

    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,c=colors, cmap='Spectral')


def hyperbolicMapper(data, colors):
    hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
                                  random_state=42).fit(data)
    plt.scatter(hyperbolic_mapper.embedding_.T[0],
                hyperbolic_mapper.embedding_.T[1],
                c=colors, cmap='Spectral')
    x = hyperbolic_mapper.embedding_[:, 0]
    y = hyperbolic_mapper.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_ ** 2, axis=1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, cmap='Spectral')
    ax.view_init(35, 80)
