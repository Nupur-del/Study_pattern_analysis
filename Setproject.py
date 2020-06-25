#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import math

datas = pd.read_csv("study_pattern_modified.csv")
data = pd.DataFrame(datas)
print(data)

data_copy = data.copy()
print(data_copy)


# In[79]:


datas.head(2)

data_copy.head(2)


# In[85]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def makeClusters(data,branch,numClusters):
    km = KModes(n_clusters=numClusters, init = 'Cao', n_init = 1,verbose=1)
    subsetDf = data.loc[data['Branch'] == branch].drop(["Internal_Exams","Study_hours_before_exam",
                                                        "Avg_hours_for_Midsem"
                                                        ,"Study_hours_before_final_exam",
                                                        "Avg_score_in_CAT","Avg_Score_in_FAT","Satisfaction",
                                                        "Stress","Difficulty_Level"],axis=1)
    subsetData = subsetDf.values
    fitClusters = km.fit_predict(subsetData)
    clusterCentroidsDf = pd.DataFrame(km.cluster_centroids_)
    clusterCentroidsDf.columns = subsetDf.columns
    silhouette_avg = silhouette_score(subsetData, fitClusters)
    print("For n_clusters =", numClusters,
          "\n The average silhouette_score is :", silhouette_avg)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    colors = cm.nipy_spectral(fitClusters.astype(int) / numClusters)
    ax2.scatter(subsetData[:, 0], subsetData[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    # Labeling the clusters
    centers = km.cluster_centroids_

    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KModes clustering on sample data "
                  "with n_clusters = %d" % numClusters),
                 fontsize=14, fontweight='bold')
    
    clustersDf = pd.DataFrame(fitClusters)
    clustersDf.columns = ['cluster_predicted']
    data_another = data_copy.reset_index()
    combinedDf = pd.concat([data_another, clustersDf], axis = 1).reset_index()
    combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)
    plt.subplots(figsize = (15,5))
    
    sns.countplot(x=combinedDf['Study_place'],order=combinedDf['Study_place'].value_counts().index,hue=combinedDf['cluster_predicted'])

    plt.show()
    return fitClusters, clusterCentroidsDf


# In[86]:


print("#######################################")
print("Branch : CSE")

clusterData = data

clusterData1_2clusters = makeClusters(clusterData,1,2)

clusterData1_2clusters[1]


# In[82]:


print("#######################################")
print("Branch : ME")

clusterData4_2clusters = makeClusters(clusterData,4,2)

clusterData4_2clusters[1]


# In[83]:


print("#######################################")
print("Branch : IT")

clusterData4_2clusters = makeClusters(clusterData,3,2)

clusterData4_2clusters[1]


# In[84]:


print("#######################################")
print("Branch : ECE,EEE")

clusterData4_2clusters = makeClusters(clusterData,2,2)

clusterData4_2clusters[1]

