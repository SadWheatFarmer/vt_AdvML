#TODO - Create a main that will use fcts from both pys.



import hierarchyClustering as hc
from som import som
import pandas as pd

##########################
################
##########################

DEBUG = False
HIERARCHICAL = True
SOM = False


YEARS = [[1970, 1979],
         [1980, 1989],
         [1990, 1999],
         [2000, 2009],
         [2010, 2017]]

# YEARS = [[1970, 1979]]
if DEBUG:
    YEARS = [YEARS[0]]

for YEAR in YEARS:
    DATA_PATH = "../data/Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
    df_data = pd.read_csv(DATA_PATH)

    if HIERARCHICAL:
        if hc.hierarchicalClustering(df_data, [YEAR[0], YEAR[1]]):
            print("Model1 (Divisive Clustering): COMPLETE")

        if hc.calcPositionConc(df_data, 'Hierarchy', [YEAR[0], YEAR[1]]):
            print("Model1 Position Extraction: COMPLETE")

    if SOM:
        if som(df_data, [YEAR[0], YEAR[1]]):
            print("Model2 (SOM Clustering): COMPLETE")

        if hc.calcPositionConc(df_data, 'SOM', [YEAR[0], YEAR[1]]):
            print("Model2 Position Extraction: COMPLETE")



