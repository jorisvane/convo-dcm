#%%

# Load in the data from the text file

import pandas as pd

AVA = pd.read_csv('C:/Users/joris/Desktop/AVA2.txt',sep=",", header = None)

def avg_ratingsdata(AVA):

    AVA = AVA.iloc[1:]

    AVA2 = AVA[[2,3,4,5,6,7,8,9,10,11]]

    AVA2.columns = [1,2,3,4,5,6,7,8,9,10]

    AVA2.reset_index(drop=True, inplace=True)

    avg_rating = []
    std_rating = []

    for i,j in AVA2.iterrows():
        j = j.astype(int)
        
        N = sum(j)
        
        totaal = j[1] * 1 + j[2] * 2 + j[3] * 3 + j[4] * 4 + j[5] * 5 + j[6] * 6 + j[7] * 7 + j[8] * 8 + j[9] * 9 + j[10] * 10
        
        a = totaal/N
        
        totaal2 = j[1] * (1-a)**2 + j[2] * (2-a)**2 + j[3] * (3-a)**2 + j[4] * (4-a)**2 + j[5] * (5-a)**2 + j[6] * (6-a)**2 + j[7] * (7-a)**2 + j[8] * (8-a)**2 + j[9] * (9-a)**2 + j[10] * (10-a)**2
        std = totaal2/N

        std_rating.append(std)
        avg_rating.append(a)

    AVA[15] = avg_rating
    AVA[16] = std_rating

    return AVA

# Select only the id's, average ratings and standard deviations

df = avg_ratingsdata(AVA)

df2 = df[[1,15,16]]

# Name the columns

df2.columns = ['id','avg rating','std rating']
print(df2)

#%%
# Plot the distribution of ratings and the standard deviation for each data point agains its average rating

import matplotlib.pyplot as plt
bins = [0,1,2,3,4,5,6,7,8,9,10]
plt.hist(df2['avg rating'], bins=bins, edgecolor="k")
plt.show()

import pandas as pd
import seaborn as sns
import numpy as np

sns.scatterplot(x="avg rating", y="std rating", data=df2)




#%%

# Take the portion of the data within the bounds that are chosen
# Standard deviation threshold: 2.5
# Average rating between 3.5 and 7.0

print(df2)

df_final = df2[(df2['avg rating'] > 3.5) & (df2['avg rating'] < 7.0) & (df2['std rating'] < 2.5)]

print(df_final)

sns.scatterplot(x="avg rating", y="std rating", data=df_final)

# %%
print(df_final.dtypes)

# %%
df_final.sort_values(by=['avg rating'], inplace=True)
print(df_final)
df_final['id'] = df_final['id'].astype(str) + '.jpg'
print(df_final)

# %%
length_slice = int(len(df_final) * 0.5)
df_upper = df_final[len(df_final)-length_slice:]
df_lower = df_final[:length_slice]

import random

df_upper['cost'] = [random.uniform(0,10) for i in range(len(df_upper))]
df_lower['cost'] = [random.uniform(0,10) for i in range(len(df_lower))]

# %%
print(df_upper)
print(df_lower)


# %%
df_upper = df_upper.sample(frac = 1)
df_lower = df_lower.sample(frac = 1)


df_upper['utility'] = 0.6 * df_upper['avg rating'] - 0.3 * df_upper['cost']
df_lower['utility'] = 0.6 * df_lower['avg rating'] - 0.3 * df_lower['cost']

print(df_upper)
print(df_lower)

# %%

# Match high with low cost and create 50/50 split in choice task

import numpy as np
index_list = np.arange(len(df_upper))
labels = ['id1', 'id2', 'ilabel', 'cost1', 'cost2', 'delta_cost', 'delta_rating']
new_df = pd.DataFrame([], columns=labels)
for index in index_list:
    up_df = df_upper.iloc[index]
    low_df = df_lower.iloc[index]
    if index > len(index_list) / 2:
        if up_df['utility'] > low_df['utility']:
            append_df = pd.DataFrame([[up_df['id'], low_df['id'], 1, up_df['cost'], low_df['cost'], up_df['cost'] - low_df['cost'], up_df['avg rating'] - low_df['avg rating']]], columns=labels)
        else:
            append_df = pd.DataFrame([[low_df['id'], up_df['id'], 1, low_df['cost'], up_df['cost'], low_df['cost'] - up_df['cost'], low_df['avg rating'] - up_df['avg rating']]], columns=labels)
    else:
        if up_df['utility'] < low_df['utility']:
            append_df = pd.DataFrame([[up_df['id'], low_df['id'], 0, up_df['cost'], low_df['cost'], up_df['cost'] - low_df['cost'], up_df['avg rating'] - low_df['avg rating']]], columns=labels)
        else:
            append_df = pd.DataFrame([[low_df['id'], up_df['id'], 0, low_df['cost'], up_df['cost'], low_df['cost'] - up_df['cost'], low_df['avg rating'] - up_df['avg rating']]], columns=labels)
    new_df = pd.concat([new_df, append_df])
# %%
print(new_df)

# %%
print(len(new_df[new_df['ilabel']==1]))
print(len(new_df))

#%%
new_df = new_df.sample(frac = 1)
print(new_df)

# %%

# Save choice data to csv file for model

new_df.to_csv('dataset.csv', header=None)
# %%
