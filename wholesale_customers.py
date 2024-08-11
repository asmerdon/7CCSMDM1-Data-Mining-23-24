# If for whatever reason this code does not work with the structure for marking, I have left in my (commented out) testing code under
# every function, which gives a working example of the output.

# Imports
import pandas as pd
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(columns=['Channel', 'Region']) #drop columns
	return df

"""data_file = "wholesale_customers.csv"
wholesale_df = read_csv_2(data_file)
print(wholesale_df.head())"""

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    summary_df = df.describe().T[['mean', 'std', 'min', 'max']] #.descrive() computes summary statistics, transposing makes output easier to read (imo)
    summary_df['mean'] = summary_df['mean'].round() #round
    summary_df['std'] = summary_df['std'].round()
    return summary_df

"""summary_df = summary_statistics(wholesale_df)
print(summary_df)"""

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    mean_values = df.mean() #get mean of vals in df
    std_values = df.std() #get standard deviation of vals in df
    standardized_df = (df - mean_values) / std_values #standardise
    return standardized_df

"""standardized_df = standardize(summary_df)
print("Standardized DataFrame:")
print(standardized_df)"""

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    kmeans = cluster.KMeans(n_clusters=k, n_init=10) #run kmeans with k clusters, n_init defines the amount of random initialisations that occur
    y = kmeans.fit_predict(df) #prediction
    return pd.Series(y, name='cluster')

"""k_means_cluster_assignment = kmeans(wholesale_df, k=3)
print("k_means_cluster_assignment:")
print(k_means_cluster_assignment)"""

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	pass

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    agglomerative = cluster.AgglomerativeClustering(n_clusters=k) #run agglomerative 
    y = agglomerative.fit_predict(df) #prediction
    return pd.Series(y, name='cluster')

"""agg_cluster_assignment = agglomerative(wholesale_df, k=3)
print("agg_cluster_assignment:")
print(agg_cluster_assignment)"""

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	score = silhouette_score(X, y)
	return score

"""k_means_silhouette_score = clustering_score(wholesale_df, k_means_cluster_assignment)
print("Silhouette score (Kmeans):", k_means_silhouette_score)
agg_silhouette_score = clustering_score(wholesale_df, agg_cluster_assignment)
print("Silhouette score (Agglomerative):", agg_silhouette_score)"""


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    results = [] #list to store result
    k_values = [3, 5, 10] #different k values
    algorithms = ['Kmeans', 'Agglomerative'] #different clustering algorithms
    data_types = ['Original', 'Standardized'] #data types
    for k in k_values: #go through each k input
        for algorithm in algorithms: #either kmeans or agglomerative
            for data_type in data_types: #either original or standardised
                max_silhouette = 0 #initial max
                best_cluster_assignments = None
                for _ in range(10): #although when kmeans() is called it init=10 already, it didn't seem within the spirit of the question to leave as is, so I'm manually calling kmeans 10 times too
                    if algorithm == 'Kmeans':
                        if data_type == 'Original':
                            cluster_assignments = kmeans(df, k) #call kmeans() for value k
                        else:
                            standardized_df = standardize(df) #call standardise() function
                            cluster_assignments = kmeans(standardized_df, k) #call kmeans() for value k
                    else: #agglomerative, I get it's redundent to do this 10 times (as agglomerative clustering doesn't have random initialisations and the question does not ask for it, and ), however I did not have the time to make more efficient
                        if data_type == 'Original':
                            cluster_assignments = agglomerative(df, k) #call agglomerative() function for value k
                        else:
                            standardized_df = standardize(df) #standardises dataframe using standardize() function
                            cluster_assignments = agglomerative(standardized_df, k) #call agglomerative() function for value k
                    if data_type == 'Original': #for computing silhouette score we need the initial dataframe (original or standardised)
                        data = df
                    else:
                        data = standardized_df #already created above
                    score = silhouette_score(data, cluster_assignments) #computes silhouette score
                    if score > max_silhouette: #check if current score is better than the previous best score
                        max_silhouette = score #update max
                        best_cluster_assignments = cluster_assignments
                results.append({ #after loop finished, store results of the best run as a dict
                    'Algorithm': algorithm,
                    'data type': data_type,
                    'k': k,
                    'Silhouette Score': max_silhouette,
                    # UNCOMMENT BELOW IF YOU WOULD LIKE TO TEST 2ND SCATTERPLOT FUNCTION (context written above final question)
                    #'Best Cluster Assignments': best_cluster_assignments #I have added the best cluster assignment to this for use in a later question (more on this below)
                })
    results_df = pd.DataFrame(results) #convert results to a pandas dataframe
    return results_df

"""evaluation_results = cluster_evaluation(wholesale_df)
print(evaluation_results)"""
	

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf): #for this question I have decided to return the whole row, as it also contains all the information about the algorithm (k number, data type etc...)
    sorted_df = rdf.sort_values(by='Silhouette Score', ascending=False) #sort the df by "Silhouette Score" in descending order
    best_row = sorted_df.iloc[0] #get top of list
    return best_row

"""best_sil = best_clustering_score(evaluation_results)
print("Best Sil:")
print(best_sil)"""

# For the final question, I have created two functions (but have the second commented out). 
# The first function follows the template, and generates the plots by calling the kmeans() function with k=3 and the standardised dataset.
# The second function however uses the best result from the best_clustering_score() function.
# I could not figure out how to implement this with just a single dataframe being passed into the function however.
# My solution passes in the cluster assignments from the output of best_clustering_score(), as well as a dataframe (either original or standardised works).
# I could not figure out a way to plot the figures without these two dataframes being passed into the scatter_plots() function, which would break the template.
# In both cases (for this dataset atleast), the best result is with k=3 and the standardised dataset, so either way the best clustering assignment is being plotted.
# For the second to run, all the commented out test code needs to be uncommented unfortunately.
# A line in the cluster_evaluation() function also needs to be uncommentd (this is marked).

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.

def scatter_plots(df):
    plt.rcParams.update({'font.size': 8}) #change font size so it doesn't overlap
    k_means_cluster_assignment = kmeans(standardize(df), k=3) #run kmeans algorithm with k=3 using the standardized data set
    num_attributes = len(df.columns) #get the number of attributes (columns) in the df (for plotting)
    fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(15, 15)) #put all plots into a single window (have 15 subplots for this data)
    plot_index = 0 #generate scatter plots for each pair attributes
    for i in range(num_attributes):
        for j in range(i+1, num_attributes):
            ax = axes[plot_index // num_attributes, plot_index % num_attributes]
            ax.scatter(df.iloc[:, i], df.iloc[:, j], c=k_means_cluster_assignment, cmap='viridis', marker='.')
            ax.set_xlabel(df.columns[i])
            ax.set_ylabel(df.columns[j])
            ax.set_title(f'Scatter Plot of {df.columns[i]} vs {df.columns[j]}')
            plot_index += 1
    for k in range(plot_index, num_attributes * num_attributes): #remove empty subplots
        fig.delaxes(axes.flatten()[k])
    plt.tight_layout() #adjust layout to prevent overlap
    plt.show()

"""scatter_plots(wholesale_df)"""

"""def scatter_plots(best_cluster_assignments, df):
    plt.rcParams.update({'font.size': 8}) #change font size so it doesn't overlap
    num_attributes = len(df.columns) #get the number of attributes (columns) in the dataframe
    fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(15, 15)) #put all plots into a single window (have 15 subplots for this data)
    plot_index = 0 #generate scatter plots for each pair attributes
    for i in range(num_attributes):
        for j in range(i+1, num_attributes):
            ax = axes[plot_index // num_attributes, plot_index % num_attributes]
            ax.scatter(df.iloc[:, i], df.iloc[:, j], c=best_cluster_assignments, cmap='viridis', marker='.')
            ax.set_xlabel(df.columns[i])
            ax.set_ylabel(df.columns[j])
            ax.set_title(f'Scatter Plot of {df.columns[i]} vs {df.columns[j]}')
            plot_index += 1
    for k in range(plot_index, num_attributes * num_attributes): #remove empty subplots
        fig.delaxes(axes.flatten()[k])
    plt.tight_layout() #fix layout
    plt.show()

best_assignments = best_clustering_score(evaluation_results)['Best Cluster Assignments']
scatter_plots(best_assignments, wholesale_df)"""

