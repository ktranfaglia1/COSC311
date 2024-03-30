'''
Kyle Tranfaglia
COSC311 - Lab03
Last updated 03/28/24
This program reads in two housing information data sets and does some statistical analysis on certain attributes
'''
import stats
import pandas as pd
from matplotlib import pyplot as plt

# Read in data 
Bejaia_Region = pd.read_csv('Bejaia_Region.csv')
SidiBel_Abbes_Region =pd.read_csv('Sidi-Bel_Abbes_Region.csv')

attributes = ["Temperature", "RH", "Ws", "Rain"]

# Task 1: Mean values of four attributes of the "Bejaia Region Dataset" for fire and no fire

# Get fire mean for each attribute 
temp_fire = stats.mean(Bejaia_Region['Temperature'][Bejaia_Region['Classes'] == 'fire   '].values)
rh_fire  = stats.mean(Bejaia_Region['RH'][Bejaia_Region['Classes'] == 'fire   '].values)
ws_fire  = stats.mean(Bejaia_Region['Ws'][Bejaia_Region['Classes'] == 'fire   '].values)
rain_fire  = stats.mean(Bejaia_Region['Rain'][Bejaia_Region['Classes'] == 'fire   '].values)

# Get not fire mean for each attribute 
temp_nofire = stats.mean(Bejaia_Region['Temperature'][Bejaia_Region['Classes'] == 'not fire   '].values)
rh_nofire = stats.mean(Bejaia_Region['RH'][Bejaia_Region['Classes'] == 'not fire   '].values)
ws_nofire = stats.mean(Bejaia_Region['Ws'][Bejaia_Region['Classes'] == 'not fire   '].values)
rain_nofire = stats.mean(Bejaia_Region['Rain'][Bejaia_Region['Classes'] == 'not fire   '].values)

# Make list of fire means and no fire means
mean_fire = [temp_fire, rh_fire, ws_fire, rain_fire]
mean_nofire = [temp_nofire, rh_nofire, ws_nofire, rain_nofire]

# Print all both fire and not fire means for each attribute
for i in range(len(attributes)):
    print(attributes[i] + "\nFire mean: " + str(round(mean_fire[i], ndigits=6)) + "\nNo Fire mean: " + str(round(mean_nofire[i], ndigits=6)))

# Set up for plot
bar_width = 0.35
index = range(len(attributes))

# Plot bar chart with both fire and no fire means
plt.figure(figsize=(10, 6))
plt.bar(index, mean_fire, bar_width, label='Fire', color='red', alpha=0.7)
plt.bar([i + bar_width for i in index], mean_nofire, bar_width, label='Not Fire', color='blue', alpha=0.7)

plt.xlabel('Attributes')
plt.ylabel('Mean Values')
plt.title('Mean Values of Attributes for Fire and Not Fire')
plt.xticks([i + bar_width / 2 for i in index], attributes)
plt.legend()

plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.bar(range(len(attributes)), mean_fire, width=0.5)
# plt.title('Mean Values for Fire')
# plt.xlabel('Attributes')
# plt.ylabel('Mean Value')
# plt.xticks(ticks=range(len(attributes)), labels=attributes)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.bar(range(len(attributes)), mean_nofire, width=0.5)
# plt.title('Mean Values for No Fire')
# plt.xlabel('Attributes')
# plt.ylabel('Mean Value')
# plt.xticks(ticks=range(len(attributes)), labels=attributes)
# plt.show()

# #Observations: Mean temperature during fire is higher than no fire, mean rh during fire is lower than no fire, mean ws during fire is slightly lower than no fire,
# and mean rain during fire is significantly lower than no fire. Therefore, we may infer that fire incidents might be associated with higher temperatures,
# occur more frequently during periods of lower relative humidity, have minor correlation to wind peed yet may be associated with higher wind speeds, and 
# occur more frequently during periods of lower rainfall.

# # Task 2: Using the " Sidi-Bel Abbes Region Dataset", calculate and show the median values of four attributes

# ffmc = df_s['FFMC'].mean()
# dmc = df_s['DMC'].mean()
# # dc = df_s['DC'].max() weird error with finding mean of DC
# isi = df_s['ISI'].mean()   

# means = [ffmc, dmc, isi]

# print(means)

# temp = df_b['Temperature'].quantile(.25)
# rh = df_b[' RH'].quantile(.25)
# ws = df_b[' Ws'].quantile(.25)
# rain = df_b['Rain '].quantile(.25)

# tf = [temp, rh, ws, rain]
# print(tf)

# temp = df_b['Temperature'].quantile(.60)
# rh = df_b[' RH'].quantile(.60)
# ws = df_b[' Ws'].quantile(.60)
# rain = df_b['Rain '].quantile(.60)

# sixty = [temp, rh, ws, rain]
# print(sixty)

# temp = df_b['Temperature'].quantile(.75)
# rh = df_b[' RH'].quantile(.75)
# ws = df_b[' Ws'].quantile(.75)
# rain = df_b['Rain '].quantile(.75)

# sf = [temp, rh, ws, rain]
# print(sf)

# temp = df_s['Temperature'].std()
# rain = df_s['Rain '].std()
# bui = df_s['BUI'].std()
# fwi = df_s['FWI'].std()

# std = [temp, rain, bui, fwi]
# print(std)

# print(stats.correlation(df_s[' RH'], df_s['Temperature']))
# print(stats.correlation(df_s[' RH'], df_s[' Ws']))
# print(stats.correlation(df_s[' RH'], df_s['Rain ']))
# print(stats.correlation(df_s[' RH'], df_s['FFMC']))
# print(stats.correlation(df_s[' RH'], df_s['DMC']))
# # print(stats.correlation(df_s[' RH'], df_s['DC'])) Weird error with this column
# print(stats.correlation(df_s[' RH'], df_s['ISI']))
# print(stats.correlation(df_s[' RH'], df_s['BUI']))
# print(stats.correlation(df_s[' RH'], df_s['FWI']))

# #strongest positive: RH and Rain
# #strongest negative: RH and ISI

# temp1 = df_b['Temperature'][df_b['Classes  '] == 'fire   '].values
# temp2 = df_b['Temperature'][df_b['Classes  '] == 'not fire   '].values

# rain1 = df_b['Rain '][df_b['Classes  '] == 'fire   '].values
# rain2 = df_b['Rain '][df_b['Classes  '] == 'not fire   '].values

# print(stats.correlation(temp1, rain1))
# print(stats.correlation(temp2, rain2))

# #correlation between temperature and rain with fire and without fire


