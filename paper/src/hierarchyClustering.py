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
## Control flags and constants
################################################################################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
import scipy.cluster.hierarchy as shc

from sklearn.metrics import calinski_harabasz_score as C_H_score
from sklearn.metrics import silhouette_score as S_score
from sklearn.metrics import davies_bouldin_score as D_B_score


def normalizeData(np_array):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np_array.tolist())

    x_normalized = normalize(x_scaled)

    return x_normalized

def modifyDataForModel(df: pd.DataFrame, INCLUDE_POS) -> pd.DataFrame:

    # Add the extra id column. BUG
    df = df.drop(columns='Unnamed: 0')

    # Remove Features
    REMOVE_FEATURES = ['ID', 'Player', 'Tm', 'Pos']
    if INCLUDE_POS:
        print("Include ONE-HOT ENCODED player POSITION in the model.")
    else:
        if len(df[df['Pos'] == 'G']) > 0:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])
        print("Remove ONE-HOT ENCODED player POSITION from the model.")

    df = df.drop(columns=REMOVE_FEATURES)

    return df


def hierarchicalClustering(df: pd.DataFrame, years: list, includePos) -> bool:
    print("---- Start Hierarchy Clustering model ----")

    df_data = modifyDataForModel(df, includePos)
    x = normalizeData(df_data.to_numpy())
    print("Data for Model Modification: COMPLETE")

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

    plt.legend('Separation of NBA Players {}-{}'.format(years[0], years[1]))
    plt.savefig('../model/Hierarchy_Dendrogram_{}-{}'.format(years[0],
                                                             years[1]))
    plt.clf()

    # TODO - Understand Cophenet
    # metric that in theory measures how well a dendrogram preserves the
    # original clustering.
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    c, coph_dists = cophenet(Z, pdist(x))
    print("Cophenetic Correlation Coefficient: {:.5f}".format(c))

    # Isolate positions per player in the clusters
    df_cluster = df[['ID', 'Year', 'Player', 'Pos', 'Cluster']]
    df_cluster.to_csv("../model/MODEL_Hierarchy_Season_Stats_{}-{}.csv".format(
        years[0],
        years[1]))

    return True


def calcPositionConc(df: pd.DataFrame, modelName, YEARS: list) -> bool:
    # TODO - (consider flipping rows and columns)
    ####################################
    # Calculate the position concentration in each cluster.
    #   Output files describing the player position concentrations in each
    #   cluster. Outputs 1) a .csv with 0-1 percentages of each position in
    #   each cluster and 2) a .png PIE chart visual of the data in output
    #   artifact 1).
    #
    # Requirements:
    #   Clusters must be labeled as 0 to x
    #   Function assumes that the player's are clustered based on 5 positions.

    col = ['Total']
    col.extend(df['Pos'].unique())

    df_conc = pd.DataFrame(columns=col)

    # i = cluster #
    # j = specific position
    fig, ax = plt.subplots(nrows=1, ncols=len(df['Pos'].unique()), squeeze=True)
    for i in range(1, len(col[1:])+1):
        df_x = df[df['Cluster'] == i]
        count = [len(df_x)]

        for j in col[1:]:
            count.append(round(len(df_x[(df_x['Pos'] == j)])/count[0], 3))

        # Publish Pie chart of concentrations
        # TIP - Use the hyperparameter 'autopct='%.1f'' to print values.
        # TODO - Do better styling https://www.pythoncharts.com/matplotlib/pie-chart-matplotlib/
        ax[i-1].set_title("Cluster {}".format(i))
        if i == 1:
            ax[i-1].pie(count[1:], labels=col[1:], normalize=True)
        else:
            ax[i-1].pie(count[1:], normalize=True)

        # Save the concentrations to publish a .csv
        df_conc = df_conc.append(
                        pd.Series(count, index=df_conc.columns),
                        ignore_index=True)

    # Publish the resulting concentrations
    df_conc.to_csv("../model/CONC_{}_Season_Stats_{}-{}.csv".format(
        modelName,
        YEARS[0],
        YEARS[1]))

    # Publish resulting PIE charts of the position concentrations
    fig.legend(title="Position Concentrations {}-{}".format(
                YEARS[0],
                YEARS[1]))
    fig.savefig("../model/PIE_{}_Season_Stats_{}-{}".format(
                modelName,
                YEARS[0],
                YEARS[1]))
    fig.clf()

    return True

##################################

def calcSilhouetteCoefficient(df_data: pd.DataFrame, df_labels: pd.DataFrame):
    score = S_score(df_data.to_numpy().tolist(),
                    df_labels.to_numpy().tolist(),
                    metric='euclidean')

    return round(score, 3)


def calcCalinskiHarabaszScore(df_data: pd.DataFrame, df_labels: pd.DataFrame):
    score = C_H_score(df_data.to_numpy().tolist(),
                      df_labels.to_numpy().tolist())

    return round(score, 3)


def calcDaviesBouldinIndex(df_data: pd.DataFrame, df_labels: pd.DataFrame):
    score = D_B_score(df_data.to_numpy().tolist(),
                      df_labels.to_numpy().tolist())

    return round(score, 3)

def reportClusterScores(df: pd.DataFrame, INCLUDE_POS):
    '''
    Various calculations of cluster tightness to judge how well the
    clustering models worked.
    1) Calinski-Harabasz Score: "Variance Ratio Criterion - a higher score
                                    relates to a model with better defined
                                    clusters."
                                Range: [0, inf]
    2) Silhouette Coefficient:  "higher score relates to a model with better
                                    defined clusters."
                                Range: [-1, 1]
    3) Davies-Bouldin Index: "Zero is the lowest possible score. Values closer
                                to zero indicate a better partition."
                                Range: [0, inf]
    REFERENCE: https://scikit-learn.org/stable/modules/clustering.html
    
    :param df: Input dataframe including ALL features, not just features used in
                modeling
    :param INCLUDE_POS: FLAG to state if a ployer's position was considered
                in modeling
    '''

    # Drop features that were not used in modeling
    # TODO - Need to find way to get the features of the dataset used in
    #  modeling to this point so that only the features used in the modeling
    #  can be used to calculate the score.
    REMOVE_FEATURES = ['Unnamed: 0', 'ID', 'Player', 'Tm', 'Pos']
    if INCLUDE_POS:
        if len(df[df['Pos'] == 'G']) > 0:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])
    REMOVE_FEATURES.extend(['Cluster'])

    df_data = df.drop(columns=REMOVE_FEATURES)
    df_labels = df['Cluster']

    tightness1 = calcCalinskiHarabaszScore(df_data, df_labels)
    print("Calinski_Harabasz_Score = {:2}".format(tightness1))

    tightness2 = calcSilhouetteCoefficient(df_data, df_labels)
    print("SilhouetteCoefficient_Score = {:2}".format(tightness2))

    tightness3 = calcDaviesBouldinIndex(df_data, df_labels)
    print("Davies-Bouldin Index = {:2}".format(tightness3))
