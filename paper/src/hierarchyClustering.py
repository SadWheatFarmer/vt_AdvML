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
    df_data = modifyDataForModel(df)
    x = normalizeData(df_data.to_numpy())
    print("Data for Model Modifcation: COMPLETE")

    # see documentation for different cluster methodologies
    # { single, complete, average, weighted, centroid, median, ward }
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Z = shc.linkage( x,
                     method='ward',
                     optimal_ordering=False
                     )


    # Documentation of .cut_tree vs .fcluster (I think .fcluster is the way
    # to go.
    # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    numClusters = 5
    clusters = shc.cut_tree(Z, n_clusters=numClusters)

    # For a specific 't' number of clusters, get a 1D vector of size=(
    # #dataPts) showing which cluster each dataPt is in. Add the cluster
    # label to the dataset.
    labels = shc.fcluster(Z, criterion='maxclust', t=numClusters)
    df['Cluster'] = labels

    ##
    # Create a visual dendrogram for the linkage data.
    # NOTE:
    #   Height between U's in a dendrogram
    #   - The distance b/w each horizontal line represents the
    #   'euclidean'/'manhatten' distance between the 'center'/'closest
    #   point'/'average' of the nearest cluster. Bigger distance means bigger
    #   differentiator of cluster.

    # Dendrogram should be cut at 17.1 to have 5 clusters
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



    # Isolate positions per player in the clusters
    df_cluster = df[['ID', 'Year', 'Player', 'Pos', 'Cluster']]
    df_cluster.to_csv("../model/MODEL_Hierarchy_Season_Stats_{}-{}.csv".format(
        YEARS[0],
        YEARS[1]))


    #TODO - Extract this into a function
    ####################################
    # Calculate the position concentration in each cluster.
    #   Output a file with the resulting concentrations
    #
    # Requirements:
    #   Clusters must be labeled as 1 to x
    #   Data used must include player Position string from the original data.
    col = ['Total', 'PG', 'SG', 'SF', 'PF', 'C']
    df_conc = pd.DataFrame(columns=col)

    for i in range(1, numClusters+1):
        df_x = df[df['Cluster'] == i]
        print("Population of Cluster {} = {}".format(i, len(df_x)))
        count = [len(df_x)]

        for j in col[1:]:
            print("Population of Cluster {} Position {} = {}".format(
                i,
                j,
                len(df_x[(df_x['Pos'] == j)])
            ))
            count.append(len(df_x[(df_x['Pos'] == j)])/count[0])

        df_conc = df_conc.append(
                        pd.Series(count, index=df_conc.columns),
                        ignore_index=True)

    df_conc.to_csv("../model/CONC_Hierarchy_Season_Stats_{}-{}.csv".format(
        YEARS[0],
        YEARS[1]))

    return True


##################################

YEARS = [2000, 2009]
DATA_PATH = "../data/Season_Stats_{}-{}.csv".format(YEARS[0], YEARS[1])
df_data = pd.read_csv(DATA_PATH)

if hierarchicalClustering(df_data, YEARS):
    print("Model1 (Divisive Clustering): COMPLETE")


