#TODO - Create a main that will use fcts from both pys.



import hierarchyClustering as hc
import pandas as pd

##########################
################
##########################

HIERARCHICAL = False
APC = True


YEARS = [[1970, 1979],
         [1980, 1989],
         [1990, 1999],
         [2000, 2009],
         [2010, 2017]]

if APC:
    YEAR = YEARS[0]



for YEAR in YEARS:
    DATA_PATH = "../data/Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
    df_data = pd.read_csv(DATA_PATH)

    if HIERARCHICAL:
        if hc.hierarchicalClustering(df_data, [YEAR[0], YEAR[1]]):
            print("Model1 (Divisive Clustering): COMPLETE")

        if hc.calcPositionConc(df_data, [YEAR[0], YEAR[1]]):
            print("Model1 Position Extraction: COMPLETE")



