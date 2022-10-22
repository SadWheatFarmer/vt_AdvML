################################################################################
#   File:   hierarchyClustering.py
#   Author: John Smutny
#   Course: ECE-5424: Advanced Machine Learning
#   Date:   10/20/2022
#   Description:
#       Hierarchical Clustering model to analyze NBA positions.
#
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

def modifyData(df: pd.DataFrame, YEARS: list, REQ_GAMES, REQ_MIN) -> \
        pd.DataFrame:
    '''
    :param df:
    :return:

    Edit the dataset in the following ways
    1) Remove features not needed for model
    2) Apply filters
        Player must have played in specific years: years = { firstYear, SecondYear }
        Player must have played in at least x games.
        Player must have played at least y minutes per game played.
    '''

    # Rename the used ID feature to ID
    df = df.rename(columns={"Unnamed: 0" : "ID"})

    # Remove Features
    REMOVE_FEATURES = ['ID', 'blanl', 'blank2', 'Player', 'Tm', 'Pos']
    df = df.drop(columns=REMOVE_FEATURES)

    # 1) Year filter
    df = df[(df['Year'] >= YEARS[0]) & (df['Year'] <= YEARS[1])]

    # 2) Game filter
    df = df[(df['G'] >= REQ_GAMES)]

    # 3) Time filter
    df = df[(df['MP'] >= REQ_GAMES * REQ_MIN)]


    # Substitute NAN values with 0 (for now) #TODO - What to do about NAN.
    df = df.replace(np.nan, 0)


    return df



def hierarchicalClustering(df: pd.DataFrame, years: list) -> bool:
    x = normalizeData(df.to_numpy())

    # see documentation for different cluster methodologies
    # { single, complete, average, weighted, centroid, median, ward }
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    dendrogram = shc.dendrogram((shc.linkage(x,
                                             method='ward',
                                             optimal_ordering=False
                                             )),
                                truncate_mode='level', p=4,
                                get_leaves=True
                                )

    # k_means = KMeans(n_clusters=5, max_iter=50, random_state=20)
    # k_means.fit(df)
    # plt.figure(figsize=(8,8))
    # plt.title('Separation of NBA Players {}-{}'.format(years[0], years[1]))
    # plt.show()

    return True


##################################

DATA_PATH = "../data/Seasons_Stats.csv"
df_data = pd.read_csv(DATA_PATH)

YEARS = [2000, 2009]
REQ_GAMES = 20
REQ_MIN = 10
df_data = modifyData(df_data, YEARS, REQ_GAMES, REQ_MIN)
print("Data Modification: COMPLETE")

if hierarchicalClustering(df_data, YEARS):
    print("Model1 (Divisive Clustering): COMPLETE")

