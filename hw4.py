'''
Name: Oudai Almustafa
Date: 19 June, 2024
ISTA 331
Description: This module implements K-Means clustering on weather data from Tucson International Airport (TIA)
for the thirty-year period 1987-2016. Functions include data loading, cleaning, processing,
and the K-Means clustering algorithm.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def is_leap_year(year):
    """
    Check if the given year is a leap year.

    Parameters:
    year (int): The year to check.

    Returns:
    bool: True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two feature vectors.

    Parameters:
    vec1 (array-like): The first feature vector.
    vec2 (array-like): The second feature vector.

    Returns:
    float: The Euclidean distance between vec1 and vec2.
    """
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def make_frame():
    """
    Create a DataFrame from the TIA 1987-2016 data CSV file and set the index to datetime objects.

    Returns:
    DataFrame: The DataFrame containing the TIA data with a datetime index.
    """
    df = pd.read_csv('TIA_1987_2016.csv')
    df.index = pd.date_range(start='1987-01-01', periods=len(df), freq='D')
    return df

def clean_dewpoint(df):
    """
    Clean the dewpoint data by replacing invalid values with the average of other years.

    Parameters:
    df (DataFrame): The DataFrame containing the TIA data.
    """
    dates_to_replace = ['2010-03-10', '2010-03-11']
    for date in dates_to_replace:
        month = int(date.split('-')[1])
        day = int(date.split('-')[2])
        avg_dewpoint = df.loc[(df.index.month == month) & (df.index.day == day) & (df.index.year != 2010), 'Dewpt'].mean()
        df.loc[date, 'Dewpt'] = avg_dewpoint


def day_of_year(date):
    """
    Calculate the day of the year for a given date, treating leap years as regular years.

    Parameters:
    date (datetime): The date to process.

    Returns:
    int: The day of the year as an integer.
    """
    if date.month == 2 and date.day == 29:
        return 366
    return date.timetuple().tm_yday if not is_leap_year(date.year) else (date.timetuple().tm_yday - 1 if date.month > 2 else date.timetuple().tm_yday)


def climatology(df):
    """
    Calculate 30-year averages of feature variables for each day of the year.

    Parameters:
    df (DataFrame): The DataFrame containing the TIA data.

    Returns:
    DataFrame: The climatology DataFrame with daily averages.
    """
    df['DayOfYear'] = df.index.dayofyear
    climatology_df = df.groupby('DayOfYear').mean()
    return climatology_df


def scale(df):
    """
    Scale the feature variables in the DataFrame to the range [0, 1].

    Parameters:
    df (DataFrame): The DataFrame to scale.

    Returns:
    None: The DataFrame is modified in place.
    """
    scaler = MinMaxScaler(copy=False)
    scaler.fit_transform(df)

def get_initial_centroids(climo_df, k):
    """
    Select initial centroids for the K-Means algorithm.

    Parameters:
    climo_df (DataFrame): The climatology DataFrame.
    k (int): The number of clusters.

    Returns:
    DataFrame: A DataFrame containing the initial centroids.
    """
    indices = [i * (len(climo_df) // k) for i in range(k)]
    return climo_df.iloc[indices].reset_index(drop=True)

def classify(centroids_df, feature_vector):
    """
    Assign a feature vector to the nearest centroid.

    Parameters:
    centroids_df (DataFrame): The centroids DataFrame.
    feature_vector (Series): A feature vector.

    Returns:
    int: The label of the nearest centroid.
    """
    distances = centroids_df.apply(lambda row: euclidean_distance(row, feature_vector), axis=1)
    return distances.idxmin()

def get_labels(df, centroids_df):
    """
    Assign each day to the nearest centroid.

    Parameters:
    df (DataFrame): The DataFrame to classify.
    centroids_df (DataFrame): The centroids DataFrame.

    Returns:
    Series: A Series mapping each day to a centroid label.
    """
    return df.apply(lambda row: classify(centroids_df, row), axis=1)

def update_centroids(df, centroids_df, labels_series):
    """
    Update the centroids based on the current labels.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    centroids_df (DataFrame): The current centroids DataFrame.
    labels_series (Series): The current labels Series.

    Returns:
    DataFrame: The updated centroids DataFrame.
    """
    for label in centroids_df.index:
        cluster_points = df[labels_series == label]
        if len(cluster_points) > 0:
            centroids_df.loc[label] = cluster_points.mean()
    return centroids_df

def k_means(df, k):
    """
    Run the K-Means algorithm to cluster the data.

    Parameters:
    df (DataFrame): The DataFrame to cluster.
    k (int): The number of clusters.

    Returns:
    DataFrame: The final centroids DataFrame.
    Series: The final labels Series.
    """
    centroids = get_initial_centroids(df, k)
    labels = get_labels(df, centroids)
    while True:
        new_centroids = update_centroids(df, centroids, labels)
        new_labels = get_labels(df, new_centroids)
        if new_labels.equals(labels):
            break
        centroids, labels = new_centroids, new_labels
    return centroids, labels

def distortion(df, labels, centroids):
    """
    Calculate the distortion of the clustering.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    labels (Series): The labels of the clusters.
    centroids (DataFrame): The centroids of the clusters.

    Returns:
    float: The distortion value.
    """
    dist = 0
    for label in centroids.index:
        cluster_points = df[labels == label]
        centroid = centroids.loc[label]
        dist += np.sum(cluster_points.apply(lambda row: euclidean_distance(row, centroid), axis=1))
    return dist

def list_of_kmeans(df, max_k):
    """
    Generate a list of k-means results for different values of k.

    Parameters:
    df (DataFrame): The DataFrame to cluster.
    max_k (int): The maximum value of k.

    Returns:
    list: A list of dictionaries containing k-means results for each k.
    """
    results = []
    for k in range(1, max_k + 1):
        centroids, labels = k_means(df, k)
        dist = distortion(df, labels, centroids)
        results.append({
            'centroids': centroids,
            'labels': labels,
            'k': k,
            'distortion': dist
        })
    return results

def extract_distortion_dict(kmeans_list):
    """
    Extract the distortion values from the k-means results.

    Parameters:
    kmeans_list (list): The list of k-means results.

    Returns:
    dict: A dictionary mapping k to distortion values.
    """
    return {result['k']: result['distortion'] for result in kmeans_list}

def plot_clusters(df, labels):
    """
    Plot clusters in a grid of scatterplots.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    labels (Series): The labels of the clusters.

    Returns:
    None: Displays the plot.
    """
    sns.pairplot(df, hue=labels)
    plt.show()



