'''
File:   main.py
Author: John Smutny
Course: ECE-5424: Advanced Machine Learning
Date:   11/19/2022
Description:
    Run data analysis of NBA Positions to visually and metrically determine
    if the NBA is becoming more position-homogenious over time, by decade.

    If desired, this program will also generate the csv file necessary to run
    the modeling if it is necessary.

    Original dataset provided by Omri Goldstein.
    https://www.kaggle.com/datasets/drgilermo/nba-players-stats?select=Seasons_Stats.csv

Input:
    1) Season_Stats.csv - Dataset from Basketball-Reference.com's
    'Total' and 'Advanced' data.
    2) Players.csv - Dataset from Kaggle (same location as INPUT 1)) that
    contains the heights and weights of all players in INPUT 1.

Output:
    Various. See each model .py file.
'''

import dataPreparation as dp
import hierarchyClustering as hc
from kMeans import runKmeans
from pca import runPCA
from som import som
import pandas as pd

##########################
################
##########################

'''
Program Control flags and constants for operation.
 Please customize these values to produce the outputs desired.

-- Flags -- 
DEBUG   - Simple program run. Run only one decade of information. Most 
            commonly used for debugging purposes.
LOAD_MODEL_DATA - Determine if the program should create the data used in 
                    modeling (see the program's two inputs) or load an already 
                    created .csv file from a previous run when this flag was 
                    set to FALSE.
                    (REQUIRES two input .csv files - see file header)
OUTPUT_FILES_FLAG - Decide if reference DataQualityReports and other csvs for 
                     independent validation should be created.
                     TRUE = output reference files to ./data/ref/
                     
INCLUDE_POS - Flag to specify if models should consider a player's 
                Position (PG, SF, C, etc) in modeling.
                TRUE = a player's position will be considered in the cluster 
                models.
                
THREE_POSITION_FLAG - Specify how many positions to consider. The traditional 
                        five positions {PG, SG, SF, PF, C} or condensed 
                        summarized positions {G, F, C}
                        TRUE = Player positions are reduced to only { G, F, C }
                     
-- File Paths --
PLAYER_PATH - File path to a dataset with player height and weight
DATA_PATH - File path to a dataset with player statistics

-- Numerics and Lists --
YEARS   - List of numeric Pairs stating what year range for a model to consider.
DQR_NON_NUMERIC_COLUMNS - List from DATA_PATH of features that are not Numeric.
                            (Used by the DataQualityReport class)
REQ_GAMES - Numeric. Filter to remove players that don't play enough games
              in a season.
REG_MIN - Numeric. Filter to remove players that don't play enough
               'minutes per game' in a season.
'''
DEBUG = False
LOAD_MODEL_DATA = True

PLAYER_PATH = "../data/input/Players.csv"
DATA_PATH = "../data/input/Seasons_Stats_1950_2022.csv"  # 1950-2022
OUTPUT_FILES_FLAG = True

HIERARCHICAL = True
SOM = True
KMEANS = True
PCA_kMEANS = False

PCA = True
VARIANCE_THRESHOLD = 0.85

# TODO - Work on Jason's commit
#   1. Apply the PCA flag as 'whether you want PCA or not' to the 3 models.
#   2. Get rid of the VARIANCE hardcodes
#   3. Move Jason's 'runPCA()' and 'pcaTransform()' functions to common.
#   4. Extract results and figures
#
#   If time
#   5. Make PIE charts bar charts

REQ_GAMES = 20
REQ_MIN = 10
INCLUDE_POS = False
THREE_POSITION_FLAG = False

DQR_NON_NUMERIC_COLUMNS = ['Unnamed: 0', 'Player', 'Tm', 'Pos',
                           'blanl', 'blank2']

YEARS = [[1971, 1980],
         [1981, 1990],
         [1991, 2000],
         [2001, 2010],
         [2011, 2020]]

if DEBUG:
    YEARS = [YEARS[0], YEARS[1]]

##########################
################
##########################

'''
** Program Execution starts HERE **
'''
# Load your own correctly formatted csv file to reduce computation time.
if LOAD_MODEL_DATA:
    df_data = pd.read_csv("../data/ref/Season_Stats_MODEL_{}-{}.csv".format(
        YEARS[0][0], YEARS[len(YEARS) - 1][1]),
        index_col=False)
else:
    df_data = dp.initialDataModification(PLAYER_PATH, DATA_PATH, YEARS,
                                         REQ_GAMES, REQ_MIN,
                                         THREE_POSITION_FLAG,
                                         DQR_NON_NUMERIC_COLUMNS,
                                         OUTPUT_FILES_FLAG)

# Create Cluster Metric dataframe placeholder to collect all metrics.
df_metrics_hierarchy = pd.DataFrame(columns=['Years', 'CHS', 'SC', 'DBI'])
df_metrics_som = pd.DataFrame(columns=['Years', 'CHS', 'SC', 'DBI'])
df_metrics_kMeans = pd.DataFrame(columns=['Years', 'CHS', 'SC', 'DBI'])
df_metrics_kMeans_pca = pd.DataFrame(columns=['Years', 'CHS', 'SC', 'DBI'])

# Begin modeling for each set of year-pairs specified.
for YEAR in YEARS:
    df_year = df_data.loc[(df_data['Year'] >= YEAR[0]) &
                          (df_data['Year'] <= YEAR[1])]

    if HIERARCHICAL:
        metrics = hc.hierarchicalClustering(df_year, [YEAR[0], YEAR[1]],
                                            INCLUDE_POS,
                                            THREE_POSITION_FLAG,
                                            PCA, VARIANCE_THRESHOLD)
        df_metrics_hierarchy.loc[len(df_metrics_hierarchy)] = metrics
        print("** Model1 (Divisive Clustering): COMPLETE\n")

    if SOM:
        metrics = som(df_year, [YEAR[0], YEAR[1]],
                      INCLUDE_POS, THREE_POSITION_FLAG,
                      PCA, VARIANCE_THRESHOLD)
        df_metrics_som.loc[len(df_metrics_som)] = metrics
        print("** Model2 (SOM Clustering): COMPLETE\n")

    if KMEANS:
        metrics = runKmeans(df_year, [YEAR[0], YEAR[1]],
                            INCLUDE_POS, THREE_POSITION_FLAG,
                            PCA, VARIANCE_THRESHOLD)
        df_metrics_kMeans.loc[len(df_metrics_kMeans)] = metrics
        print("** Model3 (KMeans): COMPLETE\n")

    if PCA_kMEANS:
        metrics = runPCA(df_year, [YEAR[0], YEAR[1]],
                         INCLUDE_POS, THREE_POSITION_FLAG,
                         VARIANCE_THRESHOLD)
        df_metrics_kMeans_pca.loc[len(df_metrics_kMeans_pca)] = metrics
        print("** Model4 (PCA KMeans): COMPLETE\n")

# Output the resulting cluster metrics to individual .csv files.
df_metrics_hierarchy.to_csv(
    '../data/output/MODEL_Metrics_Hierarchy_{}-{}.csv'.format(
        YEARS[0][0], YEARS[len(YEARS) - 1][1]),
    index=False)

df_metrics_som.to_csv(
    '../data/output/MODEL_Metrics_som_{}-{}.csv'.format(
        YEARS[0][0], YEARS[len(YEARS) - 1][1]),
    index=False)

df_metrics_kMeans.to_csv(
    '../data/output/MODEL_Metrics_kMeans_{}-{}.csv'.format(
        YEARS[0][0], YEARS[len(YEARS) - 1][1]),
    index=False)

df_metrics_kMeans_pca.to_csv(
    '../data/output/MODEL_Metrics_kMeans_pca_{}-{}.csv'.format(
        YEARS[0][0], YEARS[len(YEARS) - 1][1]),
    index=False)

'''
NOTES for later
1. When outputting metrics for clusters for DEBUG (1971-1990) vs ALL (
1971-2020), the metrics for overlapping years was not consistent. 

'''
