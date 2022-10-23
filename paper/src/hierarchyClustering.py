################################################################################
#   File:   hierarchyClustering.py
#   Author: John Smutny
#   Course: ECE-5424: Advanced Machine Learning
#   Date:   10/20/2022
#   Description:
#       Hierarchical Clustering model to analyze NBA positions.
#
#   Reference
#       Dendrograms: https://wheatoncollege.edu/wp-content/uploads/2012/08/How-to-Read-a-Dendrogram-Web-Ready.pdf
#       Hierarchical Clustering: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
## Control flags and constants
################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, normalize
import scipy.cluster.hierarchy as shc


def normalizeData(np_array):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np_array.tolist())

    x_normalized = normalize(x_scaled)

    return x_normalized

def modifyDataForModel(df: pd.DataFrame) -> pd.DataFrame:

    # Add the extra id column. BUG
    df = df.drop(columns='Unnamed: 0')

    # Remove Features
    REMOVE_FEATURES = ['Player', 'Tm', 'Pos']
    df = df.drop(columns=REMOVE_FEATURES)

    return df


def hierarchicalClustering(df: pd.DataFrame, years: list) -> bool:
    x = normalizeData(df.to_numpy())

    # see documentation for different cluster methodologies
    # { single, complete, average, weighted, centroid, median, ward }
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Z = shc.linkage( x,
                     method='ward',
                     optimal_ordering=False
                     )

    # For a specific 't' number of clusters, get a 1D vector of size=(
    # #dataPts) showing which cluster each dataPt is in. Add the cluster
    # label to the dataset.
    labels = shc.fcluster(Z, criterion='maxclust', t=5)
    df['Cluster'] = labels

    ##
    # Create a visual dendrogram for the linkage data.

    #TODO - Understand how distance plays into dendrograms. How to determine
    # the cutoff?

    dn = shc.dendrogram(Z,
                        truncate_mode='level', p=3,
                        get_leaves=True
                        )

    plt.title('Separation of NBA Players {}-{}'.format(years[0], years[1]))
    plt.show()

    # print("Dendrogram Keys = {}".format(dn.keys()))
    # print("Dendrogram icoord = {}".format(dn.get('icoord')))
    # print("Dendrogram dcoord = {}".format(dn.get('dcoord')))
    # print("Dendrogram leaves = {}".format(dn.get('leaves')))
    # print("Dendrogram ivl = {}".format(dn.get('ivl')))

    # TODO - Understand Cophenet
    # metric that in theory measures how well a dendrogram preserves the
    # original clustering.
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    c, coph_dists = cophenet(Z, pdist(x))
    print("Cophenetic Correlation Coefficient: {}".format(c))


    # k_means = KMeans(n_clusters=5, max_iter=50, random_state=20)
    # k_means.fit(df)
    # plt.figure(figsize=(8,8))
    # plt.title('Separation of NBA Players {}-{}'.format(years[0], years[1]))
    # plt.show()


    return True


##################################

YEARS = [2000, 2009]
DATA_PATH = "../data/Season_Stats_{}-{}.csv".format(YEARS[0], YEARS[1])
df_data = pd.read_csv(DATA_PATH)

df_data = modifyDataForModel(df_data)
print("Data for Model Modifcation: COMPLETE")

if hierarchicalClustering(df_data, YEARS):
    print("Model1 (Divisive Clustering): COMPLETE")


