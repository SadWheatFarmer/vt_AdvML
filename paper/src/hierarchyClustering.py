################################################################################
#   File:   hierarchyClustering.py
#   Author: John Smutny
#   Course: ECE-5424: Advanced Machine Learning
#   Date:   10/30/2022
#   Description:
#       Hierarchical Clustering model to analyze NBA positions.
#
#   Reference
#       Dendrograms: https://wheatoncollege.edu/wp-content/uploads/2012/08/How-to-Read-a-Dendrogram-Web-Ready.pdf
#       Hierarchical Clustering: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc

import lib.modelCommon as common

def modifyDataForModel(df: pd.DataFrame,
                       INCLUDE_POS_FLAG, THREE_POS_FLAG) -> pd.DataFrame:

    # Remove Features
    REMOVE_FEATURES = ['ID', 'Year', 'Player', 'Tm', 'Pos']

    # Also delete position features if they should not be used in modeling.
    if not INCLUDE_POS_FLAG:
        if THREE_POS_FLAG:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])

    df = df.drop(columns=REMOVE_FEATURES)

    return df


def hierarchicalClustering(df: pd.DataFrame, YEARS: list,
                           INCLUDE_POS, THREE_POS_FLAG) -> bool:
    print("---- Start Hierarchy Clustering model ----")

    df_data = modifyDataForModel(df, INCLUDE_POS, THREE_POS_FLAG)
    x = common.normalizeData(df_data.to_numpy())
    print("** Data for Model Modification: COMPLETE")

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
    numClusters = len(df['Pos'].unique())
    clusters = shc.cut_tree(Z,
                            n_clusters=numClusters)

    # For a specific 't' number of clusters, get a 1D vector of size=(
    # #dataPts) showing which cluster each dataPt is in. Add the cluster
    # label to the dataset.
    labels = shc.fcluster(Z,
                          criterion='maxclust',
                          t=numClusters)

    # Ensure that all labels are corrected to be in range [0, 4]
    df.loc[:, 'Cluster'] = labels - 1

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

    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=8)
    plt.legend('Separation of NBA Players {}-{}'.format(YEARS[0], YEARS[1]))
    plt.savefig('../model/Hierarchy_Dendrogram_{}-{}'.format(YEARS[0],
                                                             YEARS[1]))
    plt.clf()

    # TODO - Understand Cophenet
    # metric that in theory measures how well a dendrogram preserves the
    # original clustering.
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    c, coph_dists = cophenet(Z, pdist(x))
    print("Cophenetic Correlation Coefficient: {:.5f}".format(c))

    #####################################
    # Evaluate the Model
    # 1) Output PIE concentration charts of the clusters
    # 2) Measure the Tightness of each cluster
    common.calcPositionConc(df, "Hierarchy", YEARS, THREE_POS_FLAG)

    return common.reportClusterScores(df, YEARS, INCLUDE_POS)

