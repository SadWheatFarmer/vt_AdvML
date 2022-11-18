#TODO - Create a main that will use fcts from both pys.



import hierarchyClustering as hc
from som import som
import pandas as pd

##########################
################
##########################

DEBUG = False
INCLUDE_POS = False
HIERARCHICAL = True
SOM = True


YEARS = [[1971, 1980],
         [1981, 1990],
         [1991, 2000],
         [2001, 2010],
         [2011, 2020]]

if DEBUG:
    YEARS = [YEARS[0]]

for YEAR in YEARS:
    DATA_PATH = "../data/Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
    df_data = pd.read_csv(DATA_PATH)

    if HIERARCHICAL:
        if hc.hierarchicalClustering(df_data, [YEAR[0], YEAR[1]], INCLUDE_POS):
            print("Model1 (Divisive Clustering): COMPLETE")

        if hc.calcPositionConc(df_data, 'Hierarchy', [YEAR[0], YEAR[1]]):
            print("Model1 Position Extraction: COMPLETE")

    if SOM:
        if som(df_data, [YEAR[0], YEAR[1]], INCLUDE_POS):
            print("Model2 (SOM Clustering): COMPLETE")

        if hc.calcPositionConc(df_data, 'SOM', [YEAR[0], YEAR[1]]):
            print("Model2 Position Extraction: COMPLETE")



