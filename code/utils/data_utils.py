import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
import importlib
import sys
import os
import psutil
from sklearn.neighbors import NearestNeighbors

def reload_module(module_name):
    importlib.reload(sys.modules[module_name])

def memory_usage():
    # Get memory information
    memory_info = psutil.virtual_memory()

    # Print memory usage
    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Memory usage percentage: {memory_info.percent}%")

def make_bins(df, color_var, num_bins=10):
    """
    Create bins for a continuous variable such that each bin has roughly the same number of observations.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    color_var (str): The name of the column to bin.
    num_bins (int): The number of bins to create.
    
    Returns:
    pd.DataFrame: The dataframe with the binned variable.
    """
    # Create bins with approximately the same number of observations
    df['bin'] = pd.qcut(df[color_var], q=num_bins, duplicates='drop')
    
    # Calculate the mean of each bin
    bin_means = df.groupby('bin')[color_var].mean()
    
    # Map the original values to the mean of their respective bins
    df[color_var] = df['bin'].map(bin_means)
    
    # Drop the temporary 'bin' column
    df.drop('bin', axis=1, inplace=True)

    return df

def make_zscore_bins(df, color_var):
    """
    Create bins for a continuous variable based on z-scores.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    color_var (str): The name of the column to bin.
    
    Returns:
    pd.DataFrame: The dataframe with the binned variable.
    """
    # Calculate z-scores
    df['z_score'] = (df[color_var] - df[color_var].mean()) / df[color_var].std()
    
    # Define bins and labels
    z_score_steps = [x.round(2) for x in list(np.arange(-1, 1.2, 0.1))]
    bins = [-np.inf] + [-2] + z_score_steps + [2] + [np.inf]
    labels = ['<-2'] + [-2] + [str(x) for x in z_score_steps[:-1]] + [2] + ['>2']
    
    # Create bins
    df[color_var] = pd.cut(df['z_score'], bins=bins, labels=labels)

    
    # Drop the temporary 'z_score' column
    df.drop('z_score', axis=1, inplace=True)

    return df

def find_closest_trajectory(df, df_spaced, color_var, axis):
    """
    Finds the closest trajectory in the original dataframe to the spaced trajectory.
    
    Parameters:
    df (pd.DataFrame): The original dataframe.
    df_spaced (pd.DataFrame): The dataframe with spaced trajectories.
    color_var (str): The name of the column representing the color variable.
    axis (list): The list of axis columns.
    
    Returns:
    pd.DataFrame: The dataframe with the closest trajectories.
    """
    data = [df[df[color_var] == df[color_var].min()], df[df[color_var] == df[color_var].max()]]
    # Remove min and max from df and df_spaced
    already_assigned = [df[color_var].min(), df[color_var].max()]
    for z_score in df_spaced[color_var].unique():
        if z_score in [df[color_var].min(), df[color_var].max()]:
            continue
        # Get the coordinates for the current z_score in df_spaced
        closest_distance = np.inf
        spaced_coords = df_spaced[df_spaced[color_var] == z_score][axis].values

        # Get the coordinates for the current z_score in df
        for z in df[color_var].unique():
            if z in already_assigned:
                continue
            
            coords = df[df[color_var] == z][axis].values
            df_coords = pd.DataFrame(coords, columns=axis)

            # Compute the pairwise distances
            distance = np.diagonal(cdist(spaced_coords, df_coords)).mean()
            if distance < closest_distance:
                closest_distance = distance
                df_close = df_coords.copy()
                assigned_z = z
                df_close[color_var] = assigned_z
        
        already_assigned.append(assigned_z)
        data.append(df_close)
    data = pd.concat(data)

    return data

def space_trajectory(df, axis_snap, color_var, num_traj):
    """
    Creates a spaced trajectory between the minimum and maximum values of a color variable.
    
    Parameters:
    df (pd.DataFrame): The original dataframe.
    axis_snap (list): The list of axis columns.
    color_var (str): The name of the column representing the color variable.
    num_traj (int): The number of trajectories to create.
    
    Returns:
    pd.DataFrame: The dataframe with spaced trajectories.
    """
    min_z = df[color_var].min()
    max_z = df[color_var].max()
    max_traj = df.loc[df[color_var] == df[color_var].max(), axis_snap]
    min_traj = df.loc[df[color_var] == df[color_var].min(), axis_snap]

    data = []
    for i in range(num_traj):

        traj = min_traj.values + (max_traj.values - min_traj.values) * (i / (num_traj - 1))
        traj = pd.DataFrame(traj, columns=axis_snap)
        spaced_z = min_z + (max_z - min_z) * (i / (num_traj - 1))
        traj[color_var] = spaced_z
        data.append(traj)

    data = pd.concat(data)
    return data

def identify_variable_types(df, unique_threshold=0.05):
    """
    Identifies continuous and discrete variables in a dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    unique_threshold (float): The threshold for determining if a variable is continuous.
    
    Returns:
    tuple: A tuple containing two lists - continuous variables and discrete variables.
    """
    continuous_vars = []
    discrete_vars = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        
        # Check for continuous variables
        if (df[col].dtype == 'float' and unique_count!=2) or ((df[col].dtype == 'int') and (unique_count / total_count > unique_threshold)):
            continuous_vars.append(col)
        else:
            discrete_vars.append(col)
    return continuous_vars, discrete_vars

def plot_mean_std(df):
    """
    Plots the distribution of means and standard deviations for each column in a dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    None
    """
    # Calculate mean and standard deviation for each column
    means = df.mean()
    std_devs = df.std()

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of means
    ax[0].hist(means, bins=20, color='skyblue', edgecolor='black')
    ax[0].set_title('Distribution of Means across Columns')
    ax[0].set_xlabel('Mean')
    ax[0].set_ylabel('Frequency')

    # Histogram of standard deviations
    ax[1].hist(std_devs, bins=20, color='salmon', edgecolor='black')
    ax[1].set_title('Distribution of Standard Deviations across Columns')
    ax[1].set_xlabel('Standard Deviation')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()