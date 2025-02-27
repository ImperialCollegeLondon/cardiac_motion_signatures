import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.font_manager as font_manager
import seaborn as sns
import plotly.express as px

import numpy as np
import pandas as pd 

import math
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev, interp1d

from data_utils import make_bins, make_zscore_bins, space_trajectory, find_closest_trajectory

def plot_settings():
    """
    Sets the plot settings for consistent styling across plots.

    """
    # Add every font at the specified location
    # Download font from https://font.download/font/arial
    font_dir = [r"/home/pps21@isd.csc.mrc.ac.uk/pps21/project/mytools/arial"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
        
    # Plot settings
    disc_color = px.colors.qualitative.Plotly
    cont_color = px.colors.sequential.Turbo
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["pdf.fonttype"] = "truetype"
    plt.rcParams['figure.dpi'] = 300

    # Set all axes visible and black
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0

    # Set ticks and tick markers to be black
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3

    # Set axis titles to be black
    plt.rcParams['axes.labelcolor'] = 'black'

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Remove grid
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.color'] = 'none'

def remove_legend_duplicates(handles, labels):
    """
    Removes duplicate entries from the legend.
    
    Parameters:
    handles (list): List of legend handles.
    labels (list): List of legend labels.
    
    Returns:
    tuple: A tuple containing the unique handles and labels.
    """
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    return list(unique.values()), list(unique.keys())

def color_mapping(df, color_var):
    """
    Maps colors to the values of a specified variable.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    color_var (str): The name of the column to map colors to.
    
    Returns:
    tuple: A tuple containing a boolean indicating if the variable is continuous, the color palette, and the color normalization.
    """
    continuous_var = False
    color_palette = 'hsv'
    color_norm = None
    if color_var:
        continuous_var = (np.issubdtype(df[color_var].dtype, np.number) and len(df[color_var].unique()) > 15)
        print(df[color_var].dtype)
        if continuous_var:
            color_palette = sns.color_palette("viridis", as_cmap=True) 
            color_norm = mcolors.Normalize(vmin=df[color_var].min(), vmax=df[color_var].max())
            color_norm = mcolors.Normalize(vmin=df[color_var].quantile(0.1), vmax=df[color_var].quantile(0.9))

        else:
            unique_vals = sorted(df[color_var].unique())
            color_palette = {val: color for val, color in zip(unique_vals, sns.color_palette("hsv", len(unique_vals)))}
            color_norm = None
    else:
        color_palette = sns.color_palette("Accent", len(df.reset_index()['patient_id'].unique()))    

    return continuous_var, color_palette, color_norm

def interpolate_traj(data, axis):
    """
    Interpolates trajectory data to create a smooth curve.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing trajectory data.
    axis (list): The list of axis columns.
    
    Returns:
    pd.DataFrame: The interpolated trajectory data.
    """
    # Extract x and y coordinates
    x = data[axis[0]]
    y = data[axis[1]]
    
    # Calculate cumulative distance along the trajectory
    distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distance = np.insert(np.cumsum(distance), 0, 0)
    
    # Create interpolation functions for x and y based on cumulative distance
    interp_func_x = interp1d(cumulative_distance, x, kind='cubic')
    interp_func_y = interp1d(cumulative_distance, y, kind='cubic')
    
    # Generate new cumulative distance values for interpolation
    cumulative_distance_new = np.linspace(cumulative_distance.min(), cumulative_distance.max(), num=500)
    
    # Compute new x and y values
    x_new = interp_func_x(cumulative_distance_new)
    y_new = interp_func_y(cumulative_distance_new)
    
    # Create a new DataFrame for the interpolated data
    interpolated_data = pd.DataFrame({axis[0]: x_new, axis[1]: y_new})
    
    return interpolated_data

def draw_plot(data, axis, color_var, color_palette, color_norm, annot=False, ax_bool=None):
    """
    Draws a plot of the trajectory data.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing trajectory data.
    axis (list): The list of axis columns.
    color_var (str): The name of the column to map colors to.
    color_palette (str or dict): The color palette to use.
    color_norm (Normalize): The color normalization to use.
    annot (bool): Whether to annotate the plot.
    ax_bool (Axes): The axes to plot on.
    
    Returns:
    None
    """
    if ax_bool is None:
        ax = plt.gca()
    else:
        ax = ax_bool

    # Interpolate the data
    interpolated_data = interpolate_traj(data, axis)
    if color_var:
        interpolated_data[color_var] = data[color_var].ffill().bfill().iloc[0]

    # Plot the interpolated data
    lineplot = sns.lineplot(data=interpolated_data, x=axis[0], y=axis[1], 
                            hue=color_var if color_var else None, 
                            palette=color_palette, 
                            hue_norm=color_norm,
                            sort=False, legend='full', 
                            lw=2-1, linestyle='-',
                            ax=ax
                            )            
    
    # Draw arrows at every eighth of the trajectory
    num_points = len(interpolated_data)
    step = num_points // 8

    for i in range(step, num_points - step, step):
        x_start, y_start = interpolated_data.iloc[i][axis[0]], interpolated_data.iloc[i][axis[1]]
        x_end, y_end = interpolated_data.iloc[i + 1][axis[0]], interpolated_data.iloc[i + 1][axis[1]]
        dx, dy = x_end - x_start, y_end - y_start
        color = lineplot.get_lines()[-1].get_color()
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                     arrowprops=dict(
                                    headwidth=10/10 + 0.5, headlength=15/15 + 2,
                                     color=color, lw=1, shrinkA=0, shrinkB=0,
                                     connectionstyle='arc3,rad=0'))
    
    if ax_bool is not None:
        ax.legend().remove()

def reduce_to_barycenter(df, red_per=0.2):
    """
    Reduces the trajectory points to the barycenter by a specified percentage.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing trajectory data.
    red_per (float): The percentage to reduce to the barycenter.
    
    Returns:
    pd.DataFrame: The reduced trajectory data.
    """
    # Calculate the barycenter (centroid) of the trajectory points
    barycenter = df.mean()

    # Subtract the barycenter coordinates from each point in the trajectory
    df_centered = df - barycenter

    # Scale the centered points by the desired percentage
    df_scaled = df_centered * red_per

    # Add the barycenter coordinates back to the scaled points
    df_reduced = df_scaled + barycenter

    return df_reduced

def draw_morph(data, axis, color='black', color_var=None, spaced=False, opacity=False, ax_bool=None):
    """
    Draws a morph plot of the trajectory data.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing trajectory data.
    axis (list): The list of axis columns.
    color (str): The color to use for the plot.
    color_var (str): The name of the column to map colors to.
    spaced (bool): Whether to reduce to the barycenter.
    opacity (bool): Whether to use opacity for the plot.
    ax_bool (Axes): The axes to plot on.
    
    Returns:
    None
    """ 
    if ax_bool is None:
        ax = plt.gca()
    else:
        ax = ax_bool

    # Interpolate the data
    interpolated_data = interpolate_traj(data, axis)
    if color_var:
        interpolated_data[color_var] = data[color_var].ffill().bfill().iloc[0]

    if not opacity:
        alpha=data['opacity'].iloc[0]
    else:
        alpha=1
    thickness=0

    # Reduce to barycenter by percentage
    if spaced:
        interpolated_data = reduce_to_barycenter(interpolated_data, red_per=0.2)

    # Plot the interpolated data
    lineplot = sns.lineplot(data=interpolated_data, x=axis[0], y=axis[1], 
                            color=color,
                            lw = 2 + thickness,
                            sort=False, legend='full', 
                            linestyle='-',
                            alpha=alpha,
                            ax=ax
                            )            
    
    # Draw arrows at every eighth of the trajectory
    num_points = len(interpolated_data)
    step = num_points // 10

    for i in range(step, num_points - step, step):
        x_start, y_start = interpolated_data.iloc[i][axis[0]], interpolated_data.iloc[i][axis[1]]
        x_end, y_end = interpolated_data.iloc[i + 1][axis[0]], interpolated_data.iloc[i + 1][axis[1]]
        dx, dy = x_end - x_start, y_end - y_start
        color = lineplot.get_lines()[-1].get_color()
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(
                                    headwidth=10/3 + 0.5, headlength=15/3 + 2,
                                    color=color, lw=1, shrinkA=0, shrinkB=0,
                                    alpha=alpha,
                                    connectionstyle='arc3,rad=0'))
    
    if ax_bool is not None:
        ax.legend().remove()

    if color_var:
        ax.plot([], [], color=color, label=f'{color_var}')
        ax.legend()

def plot_trajectories(df, patient_id_range, axis, plot_mean=False, color_var=[]):
    """
    Plots the trajectories of the data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing trajectory data.
    patient_id_range (str or tuple): The range of patient IDs to include.
    axis (list): The list of axis columns.
    plot_mean (bool): Whether to plot the mean trajectory.
    color_var (list): The list of columns to map colors to.
    
    Returns:
    None
    """

    # Filter the dataframe to include only the specified patient IDs
    annot=False
    if patient_id_range=='all':
        unique_patient_ids = df.index.get_level_values('patient_id').unique()
    else:
        unique_patient_ids = df.index.get_level_values('patient_id').unique()[patient_id_range[0]:patient_id_range[1]] 
        df = df[df.index.get_level_values('patient_id').isin(unique_patient_ids)]
        if len(unique_patient_ids) < 20:
            annot=True

    # Only keep relevant columns
    if color_var:
        df = df[[axis[0], axis[1]] + [color_var]]
    else:
        df = df[[axis[0], axis[1]]]

    # Filter the dataframe to include only the specified columns
    if color_var:
        df = df[~df[color_var].isna()]

    # Define color mapping variables
    continuous_color, color_palette, color_norm = color_mapping(df, color_var)
    
    ## Plot the trajectories
    plt.figure(figsize=(10, 6))

    # Plot the mean trajectory if specified
    if plot_mean:
        # Bin the color variable if it is continuous
        if continuous_color:
            df = make_bins(df, color_var, num_bins=5)
        data_plot = df.groupby(['snap'] + ([color_var] if color_var else [])).mean().reset_index()
        [draw_plot(gp, axis, color_var, color_palette, color_norm, annot=True) for name, gp in data_plot.groupby([color_var])]

    # Plot the individual trajectories
    else:
        [draw_plot(gp, axis, color_var, color_palette, color_norm, annot) for name, gp in df.groupby(['patient_id'])]

    # plt.title('Average latent motion trajectories' if plot_mean else 'Latent motion trajectories')
    plt.xlabel(axis[0].replace('_', ' ').upper())
    plt.ylabel(axis[1].replace('_', ' ').upper())

    # Remove grids
    plt.grid(False)

    # Ensure the axis lines are visible and remove ticks
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        # spine.set_visible(True)
        spine.set_color('black')  # Set the color of the axis lines to black

    if continuous_color:
        plt.legend().remove() 
        cbar_ax = plt.gcf().add_axes([0.97, 0.15, 0.01, 0.7])  # Adjust the dimensions as needed
        mcolorbar.ColorbarBase(cbar_ax, cmap=color_palette, norm=color_norm, orientation='vertical')
        cbar_ax.set_ylabel(color_var)  # Label the colorbar with the name of the color_var
    else:
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles and labels:  # Check if there are handles and labels
            handles, labels = remove_legend_duplicates(handles, labels)
            sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
            labels, handles = zip(*sorted_handles_labels)
            legend = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            legend.set_title(color_var.replace('_', ' ').capitalize().replace('id', 'ID'))  # Set the title of the legend box

    ## When plotting centred signatures
    # plt.xlim(-4,4)
    # plt.ylim(-4,4)
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Plot dotted lines for x=0 and y=0
    # ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))
    # ax.axhline(0, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))

    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/traj_var/trajectory_{color_var}{'_mean' if plot_mean else ''}.pdf", 
    #             format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_traj_multi(df, patient_id_range, axis, plot_mean=False, color_vars=[]):
    """
    Plots multiple trajectories of the data (signature average plot)
    
    Parameters:
    df (pd.DataFrame): The dataframe containing trajectory data.
    patient_id_range (str or tuple): The range of patient IDs to include.
    axis (list): The list of axis columns.
    plot_mean (bool): Whether to plot the mean trajectory.
    color_var (list): The list of columns to map colors to.
    
    Returns:
    None
    """

    # Filter the dataframe to include only the specified patient IDs
    annot=False
    if patient_id_range=='all':
        unique_patient_ids = df.index.get_level_values('patient_id').unique()
    else:
        unique_patient_ids = df.index.get_level_values('patient_id').unique()[patient_id_range[0]:patient_id_range[1]] 
        df = df[df.index.get_level_values('patient_id').isin(unique_patient_ids)]
        if len(unique_patient_ids) < 20:
            annot=True

    # Only keep relevant columns
    if color_vars:
        df = df[[axis[0], axis[1]] + color_vars]
    else:
        df = df[[axis[0], axis[1]]]

    ## Plot the trajectories
    plt.figure(figsize=(10, 6))

    # Plot the mean trajectory
    data_plot = df[[axis[0], axis[1]]].groupby(['snap']).mean().reset_index()
    draw_morph(data_plot, axis, color='black', spaced=False, opacity=1)

    df_copy = df.copy()

    for i, color_var in enumerate(color_vars):
        # Define color mapping variables
        df = df_copy.copy()
        continuous_color, color_palette, color_norm = color_mapping(df, color_var)

        # Change palette
        unique_vals = sorted(df[color_var].unique())
        color_palette = {val: color for val, color in zip(unique_vals, sns.color_palette("hsv", len(color_vars)+1)[i:i+2])}
        color_palette[0] = 'black'
        # Plot the mean trajectory if specified
        if plot_mean:
            # Bin the color variable if it is continuous
            if continuous_color:
                df = make_bins(df, color_var, num_bins=5)
            data_plot = df.groupby(['snap'] + ([color_var] if color_var else [])).mean().reset_index()
            [draw_plot(gp, axis, color_var, color_palette, color_norm, annot=True) for name, gp in data_plot.groupby([color_var])]

        # Plot the individual trajectories
        else:
            [draw_plot(gp, axis, color_var, color_palette, color_norm, annot) for name, gp in df.groupby(['patient_id'])]

        # plt.title('Average latent motion trajectories' if plot_mean else 'Latent motion trajectories')
        plt.xlabel(axis[0].replace('_', ' ').upper())
        plt.ylabel(axis[1].replace('_', ' ').upper())

        # Remove grids
        plt.grid(False)

        # Ensure the axis lines are visible and remove ticks
        ax = plt.gca()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            # spine.set_visible(True)
            spine.set_color('black')  # Set the color of the axis lines to black

        if continuous_color:
            plt.legend().remove() 
            cbar_ax = plt.gcf().add_axes([0.97, 0.15, 0.01, 0.7])  # Adjust the dimensions as needed
            mcolorbar.ColorbarBase(cbar_ax, cmap=color_palette, norm=color_norm, orientation='vertical')
            cbar_ax.set_ylabel(color_var)  # Label the colorbar with the name of the color_var
        else:
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles and labels:  # Check if there are handles and labels
                handles, labels = remove_legend_duplicates(handles, labels)
                sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
                labels, handles = zip(*sorted_handles_labels)
                legend = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                legend.set_title(color_var.replace('_', ' ').capitalize().replace('id', 'ID'))  # Set the title of the legend box

        # Set axis limits to be consistent across average signature graphs
        x_min = -0.912
        x_max = 0.923
        # y_min = -0.561
        y_min = -0.8
        y_max = 0.699
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/traj_morph/traj_multi_{color_var}{'_mean' if plot_mean else ''}.pdf", 
    #             format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_morph(df, patient_id_range, axis, color_vars, spaced=False):
    """
    Plots the morph trajectories of the data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing trajectory data.
    patient_id_range (str or tuple): The range of patient IDs to include.
    axis (list): The list of axis columns.
    color_vars (list): The list of columns to map colors to.
    spaced (bool): Whether to reduce to the barycenter.
    
    Returns:
    None
    """
    # Filter the dataframe to include only the specified patient IDs
    annot=False
    if patient_id_range=='all':
        unique_patient_ids = df.index.get_level_values('patient_id').unique()
    else:
        unique_patient_ids = df.index.get_level_values('patient_id').unique()[patient_id_range[0]:patient_id_range[1]] 
        df = df[df.index.get_level_values('patient_id').isin(unique_patient_ids)]
        if len(unique_patient_ids) < 20:
            annot=True

    # Only keep relevant columns
    df = df[[axis[0], axis[1]] + color_vars]
    df = df.dropna(subset=color_vars)

    # Bin the variables
    for color_var in color_vars:
        df = make_zscore_bins(df, color_var)

    ## Plot the trajectories
    plt.figure(figsize=(10, 6))

    # Define color mapping variables
    color_palette = sns.color_palette("Set2", len(color_vars))

    # Plot the mean trajectory
    data_plot = df[[axis[0], axis[1]]].groupby(['snap']).mean().reset_index()
    draw_morph(data_plot, axis, color='black', spaced=spaced, opacity=1)

    # Plot z score trajectories for each variable
    for i, color_var in enumerate(color_vars):
        color = color_palette[i]

        # Convert color_var to numerical to remove infinite z scores
        data=df.copy(0)
        data[color_var] = pd.to_numeric(data[color_var], errors='coerce')
        data = data.dropna(subset=[color_var])

        # Compute trajectory for each z score
        data_plot = data[[axis[0], axis[1], color_var]].groupby(['snap'] + ([color_var] if color_var else [])).mean().reset_index()

        # Compute evenly dummy spaced trajectories
        num_traj=5
        data_spaced = space_trajectory(data_plot, [axis[0], axis[1]], color_var, num_traj=num_traj)
        data_plot = find_closest_trajectory(data_plot, data_spaced, color_var,axis)
        
        opacity_values = np.linspace(0.3, 1, num_traj)
        data_plot['opacity'] = data_plot[color_var].apply(lambda x: opacity_values[int((x - data_plot[color_var].min()) / (data_plot[color_var].max() - data_plot[color_var].min()) * (num_traj - 1))])
        [draw_morph(gp, axis, color, color_var, spaced=spaced) for name, gp in data_plot.groupby([color_var])]

    plt.xlabel(axis[0].replace('_', ' ').upper())
    plt.ylabel(axis[1].replace('_', ' ').upper())

    # Remove grids
    plt.grid(False)

    # Ensure the axis lines are visible and remove ticks
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        # spine.set_visible(True)
        spine.set_color('black')  # Set the color of the axis lines to black

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles and labels:  # Check if there are handles and labels
        handles, labels = remove_legend_duplicates(handles, labels)
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
        labels, handles = zip(*sorted_handles_labels)
        legend = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        legend.set_title(color_var.replace('_', ' ').capitalize().replace('id', 'ID'))  # Set the title of the legend box

    # Set axis limits to be consistent across signature distribution plots
    x_min = -0.912
    x_max = 0.923
    y_min = -0.561
    y_max = 0.699
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/traj_morph/traj_morph_{color_var}.pdf", 
    #             format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cluster_proportions(df, cluster_col):
    # Calculate the proportion of each cluster
    cluster_counts = df[cluster_col].value_counts(normalize=True).sort_index()
    
    # Create a horizontal stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 3))  # Adjust the height to make the bar less large vertically
    
    # Get the unique clusters and their proportions
    clusters = cluster_counts.index
    proportions = cluster_counts.values
    
    # Create a color map using hsv
    unique_vals = df[cluster_col].unique().categories.to_list()
    color_palette = {val: color for val, color in zip(unique_vals, sns.color_palette("hsv", len(unique_vals)))}
    
    # Plot the stacked bar
    left = 0
    for i, (cluster, proportion) in enumerate(zip(clusters, proportions)):
        ax.barh(0, proportion, left=left, color=color_palette[cluster], edgecolor='white', label=f'Cluster {cluster}', alpha=0.9)
        # Add text annotations
        ax.text(left + proportion / 2, 0, f'{proportion * 100:.1f}%', va='center', ha='center', color='black', fontsize=20)
        left += proportion
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clusters), title='Clusters')
    
    # Remove y-axis labels and ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Set x-axis to display percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    
    # Set title
    ax.set_title('Proportion of Each Cluster')
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/cluster_proportion.pdf", format='pdf', dpi=300)

    plt.show()

def plot_cluster_surfaces(df, axis, cluster_id, alpha=0.1):
    """
    Plot closed surfaces that approximate where each cluster is.

    Parameters:
    points (np.ndarray): 2D coordinates of points, shape (n_points, 2).
    cluster_ids (np.ndarray): Cluster IDs for each point, shape (n_points,).
    alpha (float): Transparency level for the surfaces, default is 0.3.
    """
    smoothing_factor=0.1
    # Create a DataFrame for easier manipulation
    df = df[[axis[0], axis[1], cluster_id]]

    # Get unique clusters and generate colors
    unique_vals = df[cluster_id].unique().categories.to_list()
    color_palette = {val: color for val, color in zip(unique_vals, sns.color_palette("hsv", len(unique_vals)))}

    # Plot each cluster
    plt.figure(figsize=(10, 6))
    all_lines = []
    for cluster in unique_vals:
        group = df[df[cluster_id] == cluster]
        if len(group) < 3:
            continue  # Convex hull requires at least 3 points

        # Compute the convex hull
        points = group[[axis[0], axis[1]]].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Interpolate the hull points to create a smooth curve
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
        tck, u = splprep([hull_points[:, 0], hull_points[:, 1]], s=smoothing_factor)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)

        # Plot the surface
        plt.fill(x_fine, y_fine, alpha=alpha, color=color_palette[cluster], label=f'{cluster}')

        # Plot the outline
        plt.plot(x_fine, y_fine, color=color_palette[cluster], linestyle='-', linewidth=0.5)

    plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.xlabel(axis[0].replace('_', ' ').upper())
    plt.ylabel(axis[1].replace('_', ' ').upper())
    plt.title('Cluster areas')
    plt.legend()
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/cluster_areas.pdf", format='pdf', dpi=300)
    plt.show()

def plot_average_trajectories_subplots(df, axis, cluster_id):
    """
    Plot subplots for each cluster, showing the average trajectory.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    axis (list): List containing the names of the x and y axis columns.
    cluster_id (str): Name of the column containing the cluster IDs.
    """
    # Create a DataFrame for easier manipulation
    df = df[[axis[0], axis[1], cluster_id]]

    # Get unique clusters and generate colors
    unique_vals = df[cluster_id].unique().categories.to_list()
    color_palette = {val: color for val, color in zip(unique_vals, sns.color_palette("hsv", len(unique_vals)))}

    # Calculate the number of subplots
    num_clusters = len(unique_vals)
    num_cols = math.ceil(math.sqrt(num_clusters))
    num_rows = math.ceil(num_clusters / num_cols)

    continuous_color, color_palette, color_norm = color_mapping(df, cluster_id)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    all_handles = []
    all_labels = []

    for i, cluster in enumerate(unique_vals):

        group = df[df[cluster_id] == cluster]
        data_plot = group.reset_index().groupby(['snap', cluster_id]).mean().reset_index().dropna()
        ax = axes[i]
        draw_plot(data_plot, axis, cluster_id, color_palette, color_norm, annot=True, ax_bool=ax) 

        # Collect handles and labels
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
  
        # Remove axes
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Remove duplicate handles and labels
    unique_labels_handles = dict(zip(all_labels, all_handles))
    unique_handles = list(unique_labels_handles.values())
    unique_labels = list(unique_labels_handles.keys())

    fig.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=num_cols, title=cluster_id.replace('_', ' ').capitalize())
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/cluster_avg_traj.pdf", format='pdf', dpi=300)
    plt.show()