'''
This script mesh_utils.py is a local module that contains functions related to mesh objects, 
3D objects, and 3D time objects derived from cardiac shape and motion data.
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

import os
import glob
import numpy as np
import pyvista as pv

# %matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') #png
pv.global_theme.allow_empty_mesh = True

def process_motion(df_all, n_frame):
    """
    Processes motion data by removing patients with missing information or frames, 
    and tidying up the column names.
    
    Parameters:
    df_all (pd.DataFrame): The dataframe containing motion data.
    n_frame (int): The number of frames expected for each patient.
    
    Returns:
    pd.DataFrame: The processed dataframe.
    """
    # Remove patients with missing information
    df_all = df_all.dropna(axis=0) 

    # Remove patients with missing frames out of the total number of frame
    size = df_all.groupby(level=0).size()
    df_all = df_all[df_all.index.get_level_values('patient_id').isin(size[size==n_frame].index)]

    # Remove duplicated patients
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    # Tidy column names and sort index
    df_all.columns = [x.split('_')[2] + '_' + x.split('_')[1] for i,x in enumerate(df_all.columns)]
    df_all = df_all.sort_index()

    return df_all

def remove_centroid(gp):
    """
    Moves the point cloud to the origin by subtracting the centroid coordinates.
    
    Parameters:
    gp (pd.DataFrame): The dataframe containing point cloud data.
    
    Returns:
    pd.DataFrame: The point cloud data centered at the origin.
    """
    gp = gp - gp.mean().mean()
    return gp

def find_max_distance(gp):
    """
    Finds the maximum distance of every point to the origin.
    
    Parameters:
    gp (pd.DataFrame): The dataframe containing point cloud data.
    
    Returns:
    float: The maximum distance of any point to the origin.
    """
    max_per = np.max(np.sqrt(np.sum(abs(gp.to_numpy())**2,axis=0)))
    return max_per

def divide_max_distance(gp):
    """
    Normalizes the point cloud by dividing by the maximum distance to the origin.
    
    Parameters:
    gp (pd.DataFrame): The dataframe containing point cloud data.
    
    Returns:
    pd.DataFrame: The normalized point cloud data.
    """
    max_per = np.max(np.sqrt(np.sum(abs(gp.to_numpy())**2,axis=0))).max()
    return gp/max_per

def normalise_cloud(df):
    """
    Normalizes the point cloud data within each coordinate.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing point cloud data.
    
    Returns:
    pd.DataFrame: The normalized point cloud data.
    """
    df = df.groupby(level=['Coordinate'],group_keys=False).apply(remove_centroid)
    df_max = df.groupby(level=['patient_id','snap','pos']).apply(find_max_distance)
    df = df / df_max.max()
    return df

def normalise_cloud_all(df):
    """
    Normalizes the point cloud data within each patient.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing point cloud data.
    
    Returns:
    pd.DataFrame: The normalized point cloud data.
    """
    df = df.groupby(level=['Coordinate'],group_keys=False).apply(remove_centroid)
    df_max = df.groupby(level=['patient_id']).apply(find_max_distance)
    df = df / df_max.max()
    return df

def normalise_traj2d(df, columns, normalisation=True):
    """
    Normalizes 2D trajectory data by removing the centroid and optionally normalizing within each patient.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing trajectory data.
    columns (list): The list of columns to be normalized.
    normalisation (bool): Whether to normalize within each patient.
    
    Returns:
    pd.DataFrame: The normalized trajectory data.
    """
    # Create a new multilevel index for trajectory coordinates
    df = df[columns].stack() 
    current_names = list(df.index.names)
    current_names[-1] = 'Coordinate'
    df.index.set_names(names=current_names, inplace=True)

    # Remove centroid
    df = df.groupby(level=['patient_id','Coordinate'],group_keys=False).apply(remove_centroid)

    if normalisation:
        # Normalisation within patient (divides by maximum distance of each patient)
        df = df.groupby(level=['patient_id']).apply(divide_max_distance)

    # # Normalisation within cohort
    # df_max = df.groupby(level=['patient_id','snap','pos']).apply(find_max_distance)
    # df = df / df_max.max()

    return df.unstack()

def plot_example_3d(input_3d, output_3d=np.array([])):
    """
    Plots a 3D scatter plot of input and output 3D coordinates.
    
    Parameters:
    input_3d (np.ndarray): The input 3D coordinates.
    output_3d (np.ndarray): The output 3D coordinates (optional).
    
    Returns:
    None
    """
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # ax.scatter(input_3d[i][:, 0], input_3d[i][:, 1], input_3d[i][:, 2], color=colors[i], s=0.05)
    ax.scatter(input_3d[:, 0], input_3d[:, 1], input_3d[:, 2]) #0.05 #10

    if output_3d.shape[0] > 0:
        ax.scatter(output_3d[:, 0], output_3d[:, 1], output_3d[:, 2])
        # Adding lines between points of the same index
        for j in range(len(input_3d)):
            ax.plot([input_3d[j, 0], output_3d[j, 0]], 
                    [input_3d[j, 1], output_3d[j, 1]], 
                    [input_3d[j, 2], output_3d[j, 2]], 
                    color='gray', linewidth=0.5)

    # Remove grid, axis, and legend
    ax.grid(False)
    ax.axis('off')

    plt.show()

def plot_seq_3d(input_3d, output_3d=np.array([]), skip=1):
    """
    Plots a sequence of 3D scatter plots of input and output 3D coordinates.
    
    Parameters:
    input_3d (list): A list of input 3D coordinates for each frame.
    output_3d (list): A list of output 3D coordinates for each frame (optional).
    skip (int): The number of frames to skip between plots.
    
    Returns:
    None
    """
    fig = plt.figure(figsize=(40,14))

    input_3d = input_3d[::skip]
    num_frames = len(input_3d)
    # num_frames = 1

    # Create a colormap from start_color to end_color
    cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#c22c15', '#0000FF'], N=num_frames)
    colors = [cmap(i / num_frames) for i in range(num_frames)]

    for i in range(num_frames):
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')

        # ax.scatter(input_3d[i][:, 0], input_3d[i][:, 1], input_3d[i][:, 2], color=colors[i], s=0.05)
        ax.scatter(input_3d[i][:, 0], input_3d[i][:, 1], input_3d[i][:, 2], color=colors[i], s=0.05) #0.05 #10

        if output_3d.shape[0] > 0:
            ax.scatter(output_3d[i][:, 0], output_3d[i][:, 1], output_3d[i][:, 2], color=colors[i])
            # Adding lines between points of the same index
            for j in range(len(input_3d[i])):
                ax.plot([input_3d[i][j, 0], output_3d[i][j, 0]], 
                        [input_3d[i][j, 1], output_3d[i][j, 1]], 
                        [input_3d[i][j, 2], output_3d[i][j, 2]], 
                        color='gray', linewidth=0.5)

        # Remove grid, axis, and legend
        ax.grid(False)
        ax.axis('off')

        # Change view to make x-axis perpendicular to the screen with a slight elevation
        ax.view_init(elev=0, azim=90)

        # Set aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        # Set title to be the frame number
        # ax.set_title(f'Frame {i * skip + 1}', fontsize=25, verticalalignment='top', horizontalalignment='right')

        # Set limits to the range of the data to remove empty space
        all_data = np.vstack((input_3d[i], output_3d[i])) if output_3d.shape[0] > 0 else input_3d[i]
        ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
        ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
        ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))

    # Adjust the spacing between subplots and reduce margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

    # Show plot
    # plt.savefig(r"P:\paper\motion-signatures\figures\cycle_snaps.pdf", format='pdf')

    plt.show()

def print_paper_examples():

    # Population average
    MEAN_DIR = r"P:/data/decim/motion_0.97/mean_mesh"
    vtk_files = glob.glob(os.path.join(MEAN_DIR, '*.vtk'))

    plot_seq_3d([mesh.decimate(0.97).points for mesh in pv.read(vtk_files)], skip=10)
    plot_seq_3d([mesh.decimate(0.97).points for mesh in pv.read(vtk_files)], skip=3)
