'''
File:   modelCommon.py
Author: John Smutny
Course: ECE-5424: Advanced Machine Learning
Date:   11/19/2022
Description:
    Support file to 'main.py'
    Collection of functions common to all used models such as normalization,
    scoring the resulting clusters, and creating 'Position Concentration PIE
    Charts'.
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize

from sklearn.metrics import calinski_harabasz_score as C_H_score
from sklearn.metrics import silhouette_score as S_score
from sklearn.metrics import davies_bouldin_score as D_B_score

def normalizeData(np_array):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np_array.tolist())

    x_normalized = normalize(x_scaled)

    return x_normalized

def calcPositionConc(df: pd.DataFrame, MODEL_NAME, YEARS: list, THREE_POS_FLAG):
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

    # TODO - Hardcode the order of the 'pos' fields from smallest -> largest.
    #  When INCLUDE_POS_FLAG=FALSE, avoid having the order on the pie chart
    #  be random.
    col = ['Total']

    # Add 'pos' columns for PIE chart in a specific order.
    if THREE_POS_FLAG:
        col.extend(['G', 'F', 'C'])
    else:
        col.extend(['PG', 'SG', 'SF', 'PF', 'C'])

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
        MODEL_NAME,
        YEARS[0],
        YEARS[1]))

    # Publish resulting PIE charts of the position concentrations
    fig.legend(title="Position Concentrations {}-{}".format(
                YEARS[0],
                YEARS[1]))
    fig.savefig("../model/PIE_{}_Season_Stats_{}-{}".format(
                MODEL_NAME,
                YEARS[0],
                YEARS[1]))
    fig.clf()

    print("** Model {} Position Extraction: COMPLETE".format(MODEL_NAME))


#========================================

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

def reportClusterScores(df: pd.DataFrame, YEARS: list, INCLUDE_POS):
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
    REMOVE_FEATURES = ['ID', 'Player', 'Tm', 'Pos']
    if not INCLUDE_POS:
        if len(df[df['Pos'] == 'G']) > 0:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])
    REMOVE_FEATURES.extend(['Cluster'])

    # Divide the existing dataset into FEATURES and LABELS for scoring.
    df_data = df.drop(columns=REMOVE_FEATURES)
    df_labels = df['Cluster']

    tightness1 = calcCalinskiHarabaszScore(df_data, df_labels)
    print("Calinski_Harabasz_Score = {:2}".format(tightness1))

    tightness2 = calcSilhouetteCoefficient(df_data, df_labels)
    print("SilhouetteCoefficient_Score = {:2}".format(tightness2))

    tightness3 = calcDaviesBouldinIndex(df_data, df_labels)
    print("Davies-Bouldin Index = {:2}".format(tightness3))

    return ["{}-{}".format(YEARS[0], YEARS[1]),
            tightness1, tightness2, tightness3]


