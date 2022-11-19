#TODO - Create a main that will use fcts from both pys.

import dataPreparation as dp
import hierarchyClustering as hc
from som import som
import pandas as pd

##########################
################
##########################

# TODO - Consolidate the .py files so that all of the levers are in one place.
DEBUG = False
LOAD_MODEL_DATA = False

PLAYER_PATH = "../data/input/Players.csv"
#DATA_PATH = "../data/input/Seasons_Stats.csv" #1950-2017
DATA_PATH = "../data/input/Seasons_Stats_1950_2022.csv"  #1950-2022
OUTPUT_FILES_FLAG = False

HIERARCHICAL = True
SOM = True

REQ_GAMES = 20
REQ_MIN = 10
INCLUDE_POS = True
THREE_POSITION_FLAG = False

DQR_NON_NUMERIC_COLUMNS = ['Unnamed: 0', 'Player', 'Tm', 'Pos',
                           'blanl', 'blank2']

YEARS = [[1971, 1980],
         [1981, 1990],
         [1991, 2000],
         [2001, 2010],
         [2011, 2020]]

if DEBUG:
    YEARS = [YEARS[0]]


##########################
################
##########################

# TODO - Decide if it makes more practical sense to have ONE csv that is
#  loaded instead of loading one file per year-pair.

# Load your own correctly formatted csv file to reduce computation time.
if LOAD_MODEL_DATA:
    df_data = pd.read_csv("../data/ref/Season_Stats_MODEL_{}-{}.csv".format(
                            YEARS[0][0], YEARS[len(YEARS)-1][1]))
else:
    df_data = dp.initialDataModification(PLAYER_PATH, DATA_PATH, YEARS,
                                          REQ_GAMES, REQ_MIN,
                                          THREE_POSITION_FLAG,
                                          DQR_NON_NUMERIC_COLUMNS,
                                          OUTPUT_FILES_FLAG)

for YEAR in YEARS:
    df_year = df_data[(df_data['Year'] >= YEAR[0])
                         & (df_data['Year'] <= YEAR[1])]

    if HIERARCHICAL:
        df_model = hc.hierarchicalClustering(df_year, [YEAR[0], YEAR[1]],
                                             INCLUDE_POS, THREE_POSITION_FLAG)
        print("** Model1 (Divisive Clustering): COMPLETE")

    if SOM:
        df_model = som(df_year, [YEAR[0], YEAR[1]],
                       INCLUDE_POS, THREE_POSITION_FLAG)
        print("** Model2 (SOM Clustering): COMPLETE")

