'''
File:
Author:
Date:
Desc:   Collect athletic performance information about the history of the
        Decathlon event of the Olympics and Trank & Field championships.

        Data from WorldAthletics.org
'''

'''
TODOs
1) Parse the 'EventScore' into the 10 events after getting web data
2) Consider removing the venue city location. Leave country.
3) Create a 'for' loop to process ALL pages
p1) Functionalize the code in 'webscrapping.py'
p2) Repeat steps for the women's heptathlon 
p3) Repeat for given data about athletes
'''


# Import packages (beautifulsoup and Requests) to interact with HTTP pages.
from bs4 import BeautifulSoup
import requests

# Import package helpful for data interaction
import pandas as pd

###
# Define file level definitions
webpages = [
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=1&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=2&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=3&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=4&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17'
            ]

###

########

# TODO - 3)
# Collect table information from the website
page = requests.get(webpages[0])
html_src = page.content

# Extract just the HTML Table. Remove all other HTML information
soup = BeautifulSoup(html_src, 'html.parser')
records = soup.find("table", {"class": "records-table"})

#print("Records Table = \n{}".format(records))

# Parse the table for HEADER information

th_list = records.find_all("th")
#print("Records Table HEADER = \n{}".format(th_list))
#print("Records Table HEADER size: \n{}".format(th_list))

header = []
for x in range(len(th_list)):
    text = th_list[x].text.replace(" ", "")
    text = text.replace("\n", "")

    header.append(text)

header.append(" ")
header.append("EventScore")

#TODO - 1)

df_data = pd.DataFrame(columns=header)
print(df_data.head())



# Parse the original html table for ATHLETE and PERFORMANCE information

td_list = records.find_all("td")

ITEMS_PER_ATHLETE = len(header)

athlete = [None]*ITEMS_PER_ATHLETE
#df_data.loc[len(df_data.index)] = athlete
#print(df_data.head())


for y in range(len(td_list)):
    if y%ITEMS_PER_ATHLETE == 0:
        df_data.loc[len(df_data.index)] = athlete
        print(df_data.head())

    entry = td_list[y].text.replace(" ", "")
    entry = entry.replace("\n", "")
    df_data.iat[int(y/ITEMS_PER_ATHLETE), y%ITEMS_PER_ATHLETE] = entry

    #if y < ITEMS_PER_ATHLETE:
        #print(entry)

#TODO - 2)
df_data.to_csv("decathlon_data.csv")






