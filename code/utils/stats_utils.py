import numpy as np
import pandas as pd

import math
import itertools

# import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster import hierarchy
from scipy.stats import kruskal, chi2_contingency, hypergeom, norm
import statsmodels.stats.multitest as mt
import scikit_posthocs as sp
import point_cloud_utils as pcu
from scipy.stats import zscore
from sklearn.decomposition import PCA 

from data_utils import identify_variable_types
from plot_utils import plot_settings

def perform_pca(gp, pca_n, plot=True):
    """
    Performs Principal Component Analysis (PCA) on the given data.
    
    Parameters:
    gp (pd.DataFrame): The dataframe containing the data.
    pca_n (int): The number of principal components to compute.
    plot (bool): Whether to plot the cumulative explained variance.
    
    Returns:
    pd.DataFrame: The dataframe containing the principal components.
    """
    pca = PCA(n_components=pca_n)
    data_pca = pca.fit_transform(gp)
    df_pca = pd.DataFrame(data=data_pca,columns=[f'pc_{x}' for x in range(1,pca_n+1)], index=gp.index)
    
    if plot:
        # Calculate cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio) * 100  # Convert to percentage

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(pca_n) + 1, cumulative_explained_variance, marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance (%)')
        plt.title('Cumulative Explained Variance by PCA')
        plt.grid(True, which='both', axis='y', color='lightgrey', linestyle='-', linewidth=0.5)  # Light grey horizontal grid lines

        # Set x-axis ticks to show all values incremented by 1
        x_ticks = np.arange(pca_n) + 1
        plt.xticks(x_ticks, labels=[str(x) for x in x_ticks])

        # Set y-axis ticks to show values every 5
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))

        plt.show()

    return df_pca

def perform_umap(df):
    """
    Performs Uniform Manifold Approximation and Projection (UMAP) on the given data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    pd.DataFrame: The dataframe containing the UMAP components.
    """
    # UMAP Dimensionality Reduction
    umap_model = UMAP(n_components=2, random_state=42)  # Adjust parameters as needed
    df_umap = umap_model.fit_transform(df)

    # Optionally, convert the result back to a DataFrame for easier handling
    df_umap = pd.DataFrame(df_umap, columns=['UMAP_1', 'UMAP_2'], index=df.index)

    return df_umap

def plot_mae(input, output):
    """
    Plots the Mean Absolute Error (MAE) between input and output data.
    
    Parameters:
    input (np.ndarray): The input data.
    output (np.ndarray): The output data.
    
    Returns:
    np.ndarray: The MAE percentage for each coordinate.
    """
    # Calculate MAE along axis 0,1, collapsing all other axes
    mae_diff = np.mean(np.abs(input - output), axis=(2, 3))

    min_values = np.min(input, axis=2)
    max_values = np.max(input, axis=2)
    data_range = np.mean(np.abs(max_values - min_values), axis=2)

    mae_percent = (mae_diff / data_range) * 100

    coords = ['X', 'Y', 'Z']

    plt.figure(figsize=(10,6))
    for i in range(mae_percent.shape[1]):
        sns_plot = sns.histplot(mae_percent[:, i], bins=100, stat='density', kde=True, label=f'Coordinate {coords[i]}')

        # Calculate mean and standard deviation
        mean_val = np.mean(mae_percent[:, i])
        std_val = np.std(mae_percent[:, i])

        # Get the y value of the peak of the KDE
        kde_y = sns_plot.get_lines()[i].get_data()[1]
        peak_y = np.max(kde_y)

        # Add mean ± std annotation near the peak
        plt.text(mean_val + 1, peak_y * 1, 
                 f'{mean_val:.2f} ± {std_val:.2f}', 
                 color=sns.color_palette()[i], fontsize=12, ha='center')

    plt.title('Distribution of MAE in 3 dimensions')
    plt.xlabel('MAE relative to data range (%)')
    plt.ylabel('Frequency (%)')
    plt.legend(frameon=False)  # Make legend background transparent

    # Convert y-axis to percentage
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{int(tick * 100)}%' for tick in yticks])

    plt.tight_layout()
    plt.show()

    return mae_percent

def plot_mae_4d(input, output):
    """
    Plots the Mean Absolute Error (MAE) between input and output data for 4D data.
    
    Parameters:
    input (np.ndarray): The input data.
    output (np.ndarray): The output data.
    
    Returns:
    np.ndarray: The MAE percentage for each coordinate.
    """
    # Calculate MAE along axis 0,1, collapsing all other axes
    mae_diff = np.mean(np.abs(input - output), axis=(1,2))

    min_values = np.min(input, axis=2)
    max_values = np.max(input, axis=2)
    data_range = np.mean(np.abs(max_values - min_values), axis=1)

    mae_percent = (mae_diff / data_range) * 100

    coords = ['X', 'Y', 'Z']

    plt.figure(figsize=(10,6))
    for i in range(mae_percent.shape[1]):
        sns_plot = sns.histplot(mae_percent[:, i], bins=100, stat='density', kde=True, label=f'Coordinate {coords[i]}')

        # Calculate mean and standard deviation
        mean_val = np.mean(mae_percent[:, i])
        std_val = np.std(mae_percent[:, i])

        # Get the y value of the peak of the KDE
        kde_y = sns_plot.get_lines()[i].get_data()[1]
        peak_y = np.max(kde_y)

        # Add mean ± std annotation near the peak
        plt.text(mean_val + 1, peak_y * 1, 
                 f'{mean_val:.2f} ± {std_val:.2f}', 
                 color=sns.color_palette()[i], fontsize=12, ha='center')

    plt.title('Distribution of MAE in 3 dimensions')
    plt.xlabel('MAE relative to data range (%)')
    plt.ylabel('Frequency (%)')
    plt.legend(frameon=False)  # Make legend background transparent

    # Convert y-axis to percentage
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{int(tick * 100)}%' for tick in yticks])

    plt.tight_layout()
    plt.show()

    return mae_percent

def plot_chamfer(input, output):
    """
    Plots the Chamfer distance between input and output data.
    
    Parameters:
    input (np.ndarray): The input data.
    output (np.ndarray): The output data.
    
    Returns:
    np.ndarray: The Chamfer distance for each sample.
    """
    chamfer_all = []
    for i in range(input.shape[0]):
        chamfer_snaps = []
        for j in range(input.shape[3]):
            chamfer_distance = pcu.chamfer_distance(input[i,:,:,j].T,output[i,:,:,j].T)
            chamfer_snaps.append(chamfer_distance)
        chamfer_snaps = np.mean(chamfer_snaps)
        chamfer_all.append(chamfer_snaps)
    chamfer_all = np.array(chamfer_all)

    plt.figure(figsize=(10,6))
    sns_plot = sns.histplot(chamfer_all, bins=100, stat='density', 
                            kde=True)

    # Calculate mean and standard deviation
    mean_val = np.mean(chamfer_all)
    std_val = np.std(chamfer_all)

    # Get the y value of the peak of the KDE
    kde_y = sns_plot.get_lines()[0].get_data()[1]
    peak_y = np.max(kde_y)

    # Add mean ± std annotation near the peak
    plt.text(mean_val + 1, peak_y * 1, 
                f'{mean_val:.2f} ± {std_val:.2f}', 
                color=sns.color_palette()[0], fontsize=12, ha='center')

    plt.title('Distribution of Chamfer distance')
    plt.xlabel('Chamfer distance')
    plt.ylabel('Frequency (%)')
    plt.legend(frameon=False)  # Make legend background transparent

    # Convert y-axis to percentage
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in yticks])

    plt.tight_layout()
    plt.show()

    return chamfer_all

def plot_chamfer_4d(input, output):
    """
    Plots the Chamfer distance between input and output data for 4D data.
    
    Parameters:
    input (np.ndarray): The input data.
    output (np.ndarray): The output data.
    
    Returns:
    np.ndarray: The Chamfer distance for each sample.
    """
    chamfer_all = []
    for i in range(input.shape[0]):
        chamfer_snaps = []
        for j in range(input.shape[1]):
            chamfer_distance = pcu.chamfer_distance(input[i,j,:,:],output[i,j,:,:])
            chamfer_snaps.append(chamfer_distance)
        chamfer_snaps = np.mean(chamfer_snaps)
        chamfer_all.append(chamfer_snaps)
    chamfer_all = np.array(chamfer_all)

    plt.figure(figsize=(10,6))
    sns_plot = sns.histplot(chamfer_all, bins=100, stat='density', 
                            kde=True)

    # Calculate mean and standard deviation
    mean_val = np.mean(chamfer_all)
    std_val = np.std(chamfer_all)

    # Get the y value of the peak of the KDE
    kde_y = sns_plot.get_lines()[0].get_data()[1]
    peak_y = np.max(kde_y)

    # Add mean ± std annotation near the peak
    plt.text(mean_val + 1, peak_y * 1, 
                f'{mean_val:.2f} ± {std_val:.2f}', 
                color=sns.color_palette()[0], fontsize=12, ha='center')

    plt.title('Distribution of Chamfer distance')
    plt.xlabel('Chamfer distance')
    plt.ylabel('Frequency (%)')
    plt.legend(frameon=False)  # Make legend background transparent

    # Convert y-axis to percentage
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in yticks])

    plt.tight_layout()
    plt.show()

    return chamfer_all

def plot_corr_heatmap(df):
    """
    Plots a heatmap of the correlation matrix of the given dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    None
    """
    # Step 1: Compute the Absolute Correlation Matrix
    corr = df.corr().abs()

    # Step 2: Perform Hierarchical Clustering
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=corr.columns.tolist(), no_plot=True)
    order = dendro['ivl']

    # Step 3: Reorder the DataFrame based on the clustering
    df_latent_ordered = df[order]

    # Step 4: Plot the Reordered Absolute Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_latent_ordered.corr().abs(), cmap='coolwarm', annot=False, square=True, cbar_kws={'shrink': .5})
    plt.title('Correlation Matrix - Grouped with Hierarchical Clustering', fontsize=16)

    plt.ylabel('Latent vector')
    plt.xlabel('Latent vector')

    # Add title to the colorbar
    cbar = plt.gcf().axes[-1]  # Get the colorbar axis
    cbar.set_ylabel('Correlation coefficient (absolute value)', rotation=270, labelpad=15)

    plt.show()

def run_test_by_cluster(df, cluster_col):
    """
    Runs statistical tests by cluster and returns the results.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    cluster_col (str): The name of the column representing the clusters.
    
    Returns:
    tuple: A tuple containing the statistics dataframe, expected values dataframe, eta squared values, and Cramer's V values.
    """ 
    continuous_vars, discrete_vars = identify_variable_types(df) 
    
    p_values = {}
    eta_square = {}
    exp = []
    cramerV = {}
    for var in continuous_vars:
        groups = [group[var].dropna() for name, group in df.groupby(cluster_col)]
        results = kruskal(*groups)
        eta_square[var] = (results.statistic - len(groups) + 1) / (len(df) - len(groups))
        p_values[var] = {'p_value': results.pvalue, 'test': 'kruskal'}

    for var in discrete_vars:
        df_var = df[~df[var].isna()]
        contingency_table = pd.crosstab(df_var.index.get_level_values(cluster_col), df_var[var])
        chi2_result = chi2_contingency(contingency_table)

        # Cramer's V or effect size
        X2 = chi2_result[0] 
        N = np.sum(contingency_table.values) 
        minimum_dimension = min(contingency_table.shape)-1
        cramerV[var] = np.sqrt((X2/N) / minimum_dimension) 
  
        p_values[var] = {'p_value': chi2_result[1], 'test': 'chi-square'}

        df_exp = pd.DataFrame(chi2_result[3], columns=contingency_table.columns, index=contingency_table.index)
        df_exp = df_exp.reset_index().melt(id_vars='row_0', var_name='value', value_name='expected_value')
        df_exp.rename(columns={'row_0': cluster_col}, inplace=True)
        df_exp['variable'] = var
        exp.append(df_exp)

    df_stats = pd.DataFrame.from_dict(p_values, orient='index')
    df_stats = df_stats.reset_index().rename(columns={'index': 'variable'})

    # Apply Benjamini-Hochberg correction
    df_stats['corrected_p_value'] = mt.multipletests(df_stats['p_value'], alpha=0.05, method='fdr_bh')[1]

    df_stats = df_stats.sort_values(by='corrected_p_value', ascending=True)
    
    return df_stats, pd.concat(exp), eta_square, cramerV

def dunn_test(df, var, cluster_col):
    """
    Performs Dunn's test for multiple comparisons.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    var (str): The variable to test.
    cluster_col (str): The name of the column representing the clusters.
    
    Returns:
    pd.DataFrame: The dataframe containing the Dunn's test results.
    """
    # Get unique clusters and sort them
    clusters_ord = sorted(df.index.get_level_values(cluster_col).unique())
    
    # Group data by clusters in the sorted order
    obs_var = [df[df.index.get_level_values(cluster_col) == cluster][var] for cluster in clusters_ord]
    
    df_dunn = sp.posthoc_dunn(obs_var)
    df_dunn = df_dunn.reset_index().rename(columns={'index':'cluster_1'})
    df_dunn = df_dunn.melt('cluster_1', var_name='cluster_2', value_name='p_value')
    df_dunn[['cluster_1', 'cluster_2']] = df_dunn[['cluster_1', 'cluster_2']] - 1
    df_dunn = df_dunn[df_dunn.p_value < 1]
    df_dunn = df_dunn[~df_dunn.apply(frozenset, axis=1).duplicated()]
    df_dunn['variable'] = var
    df_dunn = df_dunn[['variable', 'cluster_1', 'cluster_2', 'p_value']]

    return df_dunn

def hypergeom_success(df, var, value, cluster_col, cluster):
    """
    Performs a hypergeometric test for a specific value in a cluster.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    var (str): The variable to test.
    value (any): The value to test.
    cluster_col (str): The name of the column representing the clusters.
    cluster (any): The cluster to test.
    
    Returns:
    list: A list containing the variable, value, cluster, p-value, and odds ratio.
    """
    value_mask = df[var] == value
    cluster_mask = df.index.get_level_values(cluster_col) == cluster

    M = len(df)  # Total number of observations
    n = value_mask.sum()  # Successes in population (having the specific value)
    N = cluster_mask.sum()  # Sample size (size of the current cluster)
    k = (value_mask & cluster_mask).sum()  # Successes in sample (having the value in the cluster)
    p_value = 1 - hypergeom.cdf(k-1, M, n, N)

    OR = (k * (M - N - n + k)) / ((n - k) * (N - k)) # AD/BC

    return [var, value, cluster, p_value, OR]

def hypergeom_test(df, var, cluster_col):
    """
    Performs hypergeometric tests for all unique values of a variable across all clusters.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    var (str): The variable to test.
    cluster_col (str): The name of the column representing the clusters.
    
    Returns:
    pd.DataFrame: The dataframe containing the hypergeometric test results.
    """
    results = [hypergeom_success(df, var, value, cluster_col, cluster) for value in df[var].unique() for cluster in df.index.get_level_values(cluster_col).unique()]
    df_hgeom = pd.DataFrame(results, columns=['variable', 'value', cluster_col, 'p_value', 'odds_ratio'])

    return df_hgeom

def convert_pvalue_to_asterisks(pvalue):
    """
    Converts a p-value to a string of asterisks for significance levels.
    
    Parameters:
    pvalue (float): The p-value to convert.
    
    Returns:
    str: A string of asterisks representing the significance level.
    """
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def posthoc_tests(df, df_stats, cluster_col):
    """
    Performs posthoc tests (Dunn's test and hypergeometric test) for multiple comparisons.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    df_stats (pd.DataFrame): The dataframe containing the statistical test results.
    cluster_col (str): The name of the column representing the clusters.
    
    Returns:
    tuple: A tuple containing the Dunn's test results and hypergeometric test results.
    """
    continuous_var = df_stats.loc[df_stats.test == 'kruskal', 'variable'].values
    discrete_vars = df_stats.loc[df_stats.test == 'chi-square', 'variable'].values

    df_dunn = pd.DataFrame()
    df_hgeom = pd.DataFrame()

    if continuous_var.size > 0:
        df_dunn = pd.concat([dunn_test(df, var, cluster_col) for var in continuous_var], axis=0).sort_values(by='p_value', ascending=True)
    if discrete_vars.size > 0:
        df_hgeom = pd.concat([hypergeom_test(df, var, cluster_col) for var in discrete_vars], axis=0).sort_values(by='p_value', ascending=True)

    if not df_dunn.empty or not df_hgeom.empty:
        pvalues = []
        if not df_dunn.empty:
            pvalues.extend(df_dunn.p_value.values)
        if not df_hgeom.empty:
            pvalues.extend(df_hgeom.p_value.values)
        
        pvalue_bh = mt.multipletests(pvalues, alpha=0.05, method='fdr_bh')[1]
        
        cut_off = df_dunn.shape[0] if not df_dunn.empty else 0
        if not df_dunn.empty:
            df_dunn['p_value_bh'] = pvalue_bh[:cut_off]
            df_dunn = df_dunn.sort_values(by='p_value_bh', ascending=True)
            df_dunn = df_dunn[df_dunn.p_value_bh < 0.05]
            df_dunn['asterisk'] = df_dunn['p_value_bh'].apply(convert_pvalue_to_asterisks)
        if not df_hgeom.empty:
            df_hgeom['p_value_bh'] = pvalue_bh[cut_off:]
            df_hgeom = df_hgeom.sort_values(by='p_value_bh', ascending=True)
            df_hgeom = df_hgeom[df_hgeom.p_value_bh < 0.05]
            df_hgeom['asterisk'] = df_hgeom['p_value_bh'].apply(convert_pvalue_to_asterisks)

    return df_dunn, df_hgeom

def q1_diff(x):
    return x.median() - x.quantile(0.25)

def q3_diff(x):
    return x.quantile(0.75) - x.median()

def create_summary_continuous(df, df_dunn, cluster_col):
    """
    Creates a summary of continuous variables by cluster.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    df_dunn (pd.DataFrame): The dataframe containing the Dunn's test results.
    cluster_col (str): The name of the column representing the clusters.
    
    Returns:
    pd.DataFrame: The summary dataframe with median, Q1 difference, and Q3 difference.
    """
    ordered_vars = pd.Categorical(df_dunn['variable'].unique(), categories=df_dunn['variable'].unique(), ordered=True)
    df_mean_iqr = df[list(ordered_vars.categories)]
    sum_cont = df_mean_iqr.groupby(cluster_col).agg(['median', q1_diff, q3_diff])
    return sum_cont

def calculate_median_ci(data, confidence=0.95):
    """
    Calculates the median and confidence interval for the median.
    
    Parameters:
    data (np.ndarray): The data array.
    confidence (float): The confidence level for the interval.
    
    Returns:
    tuple: A tuple containing the median, lower bound, and upper bound of the confidence interval.
    """
    # Remove NaN values
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    m = np.nanmedian(data)
    se = 1.253 * np.nanstd(data) / np.sqrt(n)  # Standard error of the median
    h = se * norm.ppf((1 + confidence) / 2)
    return m, m - h, m + h

def assign_letters(df_dunn, variable):
    """
    Assigns letters to clusters based on statistical significance.
    
    Parameters:
    df_dunn (pd.DataFrame): The dataframe containing the Dunn's test results.
    variable (str): The variable to assign letters for.
    
    Returns:
    dict: A dictionary mapping clusters to letters.
    """
    letters = itertools.cycle('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    clusters = pd.unique(df_dunn[['cluster_1', 'cluster_2']].values.ravel('K'))
    cluster_letters = {cluster: '' for cluster in clusters}
    
    for cluster in clusters:
        if cluster_letters[cluster] == '':
            current_letter = next(letters)
            cluster_letters[cluster] += current_letter
            for other_cluster in clusters:
                if other_cluster != cluster:
                    # Check if there is a significant difference between the clusters
                    if not df_dunn[(df_dunn['variable'] == variable) & 
                                   (((df_dunn['cluster_1'] == cluster) & (df_dunn['cluster_2'] == other_cluster)) |
                                    ((df_dunn['cluster_1'] == other_cluster) & (df_dunn['cluster_2'] == cluster)))].empty:
                        cluster_letters[other_cluster] += current_letter

    # Align letters vertically by adding spaces
    max_length = max(len(letters) for letters in cluster_letters.values())
    for cluster in cluster_letters:
        aligned_letters = [' '] * max_length
        for i, letter in enumerate(cluster_letters[cluster]):
            aligned_letters[i] = letter
        cluster_letters[cluster] = ''.join(aligned_letters)
    
    return cluster_letters

def plot_summary_continuous(df, df_dunn, cluster_col, eta_square, df_mapping, z_score=False):
    """
    Plots a summary of continuous variables by cluster.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    df_dunn (pd.DataFrame): The dataframe containing the Dunn's test results.
    cluster_col (str): The name of the column representing the clusters.
    eta_square (dict): A dictionary containing eta squared values for each variable.
    df_mapping (pd.DataFrame): The dataframe containing variable mappings.
    
    Returns:
    None
    """
    df_dunn = pd.merge(df_dunn, df_mapping, left_on='variable', right_on='field_var', how='left')
    df_dunn = df_dunn.sort_values(by='category')

    plot_settings()
    sum_cont = create_summary_continuous(df, df_dunn, cluster_col)
    palette = sns.color_palette("hsv", n_colors=sum_cont.index.nunique())

    # Summary plots
    num_plots = len(sum_cont.columns.levels[0])
    coln = math.ceil(math.sqrt(num_plots))
    coln=4
    rown = math.ceil(num_plots / coln)

    # Calculate the number of clusters
    num_clusters = df.index.get_level_values(cluster_col).nunique()

    # Adjust the figure size based on the number of clusters
    base_height_per_cluster = 1.5  # Base height per cluster
    base_width_per_plot = 3  # Base width per plot
    base_height_per_plot = 2  # Base height per plot

    fig_width = coln * base_width_per_plot
    fig_height = rown * base_height_per_plot + num_clusters * base_height_per_cluster

    fig, axs = plt.subplots(rown, coln, figsize=(fig_width, fig_height))
    fig.tight_layout(pad=30)
    axs = axs.flatten()

    ci_all = pd.DataFrame()
    for i, col in enumerate(sum_cont.columns.levels[0]):

        col_data = df[col].reset_index()
        if z_score==True:
            col_data[col] = zscore(col_data[col], nan_policy='omit')
        ci_data = col_data.groupby(cluster_col)[col].apply(calculate_median_ci).apply(pd.Series)
        ci_data.columns = ['median', 'ci_lower', 'ci_upper']
        ci_data = ci_data.reset_index()

        axs[i].errorbar(
            x=ci_data['median'], y=ci_data[cluster_col],
            xerr=[ci_data['median'] - ci_data['ci_lower'], ci_data['ci_upper'] - ci_data['median']],
            capsize=0,
            elinewidth=2,
            ls='None',
            ecolor=palette,
        )
        axs[i].scatter(x=ci_data['median'], y=ci_data[cluster_col], s=70, marker="o", color=palette)
        
        axs[i].set_title(
            f"{df_mapping.loc[df_mapping.field_var==col,'field'].values[0]}", 
            fontsize=10,
            loc='left',
            multialignment='left'
        )

        # Calculate the maximum absolute value for setting the x-axis limits
        max_abs_value = max(abs(ci_data['ci_lower'].min()), abs(ci_data['ci_upper'].max()))*1.05

        # Center the x-axis around 0 with limits based on the maximum absolute value
        axs[i].set_xlim(-max_abs_value, max_abs_value)

        # Add a vertical dotted line at 0
        axs[i].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

        axs[i].set_title(
            f"{df_mapping.loc[df_mapping.field_var==col,'field'].values[0]}\n$\eta^2$: {eta_square[col]:.2f}",  # Add eta_square[col] on the line underneath
            fontsize=8,
            loc='left',
            multialignment='left'
        )
        
        # Assign letters to clusters based on statistical significance
        # Add letters to the right side of the plot
        cluster_letters = assign_letters(df_dunn, col)
        clusters = pd.unique(df_dunn[['cluster_1', 'cluster_2']].values.ravel('K'))
        xlim = axs[i].get_xlim()[1]
        range_x = axs[i].get_xlim()[1] - axs[i].get_xlim()[0]
        for cluster in clusters:
            letter = cluster_letters[cluster]
            axs[i].text(xlim + range_x*0.05, cluster, letter, va='center', ha='left', fontsize=10)

        # Save data for further analysis
        ci_data['variable'] = col
        ci_all = pd.concat([ci_all, ci_data])

    for ax in axs:
        ax.set_ylabel('Cluster', fontsize=12)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.xaxis.label.set_color('black')
        ax.tick_params('both', length=6, width=1, which='major', colors='black', labelsize=10)
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=3)
        ax.xaxis.label.set_visible(False)

        # Increase margins to add space between the content and the axes
        ax.margins(y=0.3, x=0.05)  # Adjust the margins as needed

    for j in range(i+1,rown*coln):
        fig.delaxes(axs[j])

    plt.tight_layout()
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/continuous_var.pdf", format='pdf', dpi=300)
    plt.show()
    return ci_all

def plot_zscore(df, df_dunn, cluster_col, df_mapping):
    """
    Plots a heatmap of z-scores for significantly associated features by cluster.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    df_dunn (pd.DataFrame): The dataframe containing the Dunn's test results.
    cluster_col (str): The name of the column representing the clusters.
    df_mapping (pd.DataFrame): The dataframe containing variable mappings.
    
    Returns:
    None
    """
    df = df[df.columns.intersection(df_dunn['variable'].unique())]
    sorted_columns = df_mapping.set_index('field_var').loc[df.columns, 'category'].sort_values().index.values
    df = df[sorted_columns]
    df = df.rename(columns=df_mapping.set_index('field_var')['field'])

    # Compute z-scores by cluster
    # zscores = df.apply(lambda x: x.apply(zscore, nan_policy='omit'))
    zscores = zscore(df, nan_policy='omit')

    # Compute average z-scores for each cluster for each variable
    avg_zscores = zscores.groupby(cluster_col).mean()

    # Prepare data for heatmap
    avg_zscores = avg_zscores.reset_index()
    avg_zscores = avg_zscores.melt(id_vars=[cluster_col], var_name='variable', value_name='zscore')
    heatmap_data = avg_zscores.pivot(index='variable', columns=cluster_col, values='zscore')
    heatmap_data = heatmap_data.loc[df.columns]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, cmap='coolwarm', center=0, cbar_kws={'shrink': 0.3, 'label': 'Z-Score'}, square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8) 
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Clusters')
    plt.title('Average Z-Score of significantly associated features by cluster')

    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/zscore_selection.pdf", format='pdf', dpi=300)
    plt.show()

def plot_latent_heatmap(df_vectors, df_clin, df_mapping):
    """
    Plots a heatmap of the correlation matrix between latent vectors and subject features.
    
    Parameters:
    df_vectors (pd.DataFrame): The dataframe containing the latent vectors.
    df_clin (pd.DataFrame): The dataframe containing the clinical data.
    df_mapping (pd.DataFrame): The dataframe containing variable mappings.
    
    Returns:
    None
    """
    # Heatmap between latent vectors and subject features
    sorted_columns = df_mapping.set_index('field_var').loc[df_clin.columns, 'category'].sort_values().index.values
    df_clin = df_clin[sorted_columns]
    df_clin = df_clin.rename(columns=df_mapping.set_index('field_var')['field'])
    numerical_columns = df_clin.select_dtypes(include=['number']).columns

    df_vectors.columns = df_vectors.columns.str.replace('pc_', 'PC ')

    # Create subplots
    fig = plt.figure(figsize=(30, 20))

    merged_df = df_vectors.join(df_clin[numerical_columns], how='inner')

    # Calculate the correlation matrix between df_latent columns and numerical columns of df_clin
    correlation_matrix = merged_df[df_vectors.columns].apply(merged_df[numerical_columns].corrwith)

    # Plot the correlation matrix using a heatmap
    ax = sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.1, 'label': 'Correlation Coefficient'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10) 
    plt.xticks(rotation=45, fontsize=8) 
    plt.yticks(fontsize=12)
    plt.xlabel('Latent vectors')
    plt.title('Correlation matrix between latent space vectors and subject features', fontsize=16)
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/heatmap_latent_space_features.pdf", format='pdf', dpi=300)
    plt.show()

def plot_grid_by_cluster(df, df_hgeom, cluster_col, df_mapping):
    """
    Plots a grid of significant associations between clusters and discrete features.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    df_hgeom (pd.DataFrame): The dataframe containing the hypergeometric test results.
    cluster_col (str): The name of the column representing the clusters.
    df_mapping (pd.DataFrame): The dataframe containing variable mappings.
    
    Returns:
    None
    """
    df_hgeom = pd.merge(df_hgeom, df_mapping, left_on='variable', right_on='field_var', how='inner')
    df_hgeom = df_hgeom.sort_values(by='category')

    # Temporary to remove double entries
    df_hgeom = df_hgeom.drop_duplicates(subset=['variable', cluster_col])

    df = df.reset_index()
    df_hgeom['p_value_bh'] = df_hgeom['p_value_bh'].replace(0, 10**(-15))
    df_hgeom['log_pvalue'] = -np.log10(df_hgeom['p_value_bh'])

    # Step 1: One-Hot Encode Non-Binary Variables
    binary_df_hgeom = pd.DataFrame()

    for _, row in df_hgeom.iterrows():
        var = row['variable']
        value = row['value']
        if value not in [0, 1]:
            new_var = f"{var}_{value}"
            temp_df = row.copy()
            temp_df['variable'] = new_var
            temp_df['field'] = f"{row['field']}: {value}"
            temp_df['value'] = 1
            temp_df['one_hot_encoded'] = True  # Add binary flag
            binary_df_hgeom = pd.concat([binary_df_hgeom, temp_df.to_frame().T], ignore_index=True)
        else:
            temp_df = row.copy()
            temp_df['one_hot_encoded'] = False  # Add binary flag
            binary_df_hgeom = pd.concat([binary_df_hgeom, temp_df.to_frame().T], ignore_index=True)

    # Step 2: Create a Fixed Grid
    # clusters = sorted(df[cluster_col].unique())
    clusters = df[cluster_col].unique().categories.to_list()
    significant_vars = binary_df_hgeom['variable'].unique()
    significant_names = binary_df_hgeom['field'].unique()

    fig, ax = plt.subplots(figsize=(50,40), facecolor='none', edgecolor='none')
    ax.set_facecolor('none')
    # Create a color palette for the clusters
    palette = sns.color_palette("hsv", len(clusters))
    cluster_colors = {cluster: palette[i] for i, cluster in enumerate(clusters)}

    # Enable and customize the grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.1, color='gray')
    ax.set_axisbelow(True)
    
    # Turn off the borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Draw the grid
    for i, cluster in enumerate(clusters):
        for j, var in enumerate(significant_vars):
            ax.scatter(i, j, color='white', edgecolor='white', s=0.1)

    # Step 3: Iterate over df_hgeom
    for cluster in clusters:
        cluster_data = df[df[cluster_col] == cluster]
        
        var_data = binary_df_hgeom[(binary_df_hgeom[cluster_col] == cluster)]
            
        for _, row in var_data.iterrows():
            if row['odds_ratio'] in ['inf', float('inf')]:
                continue
            variable = row['variable']
            value = row['value']
            variable_org = variable
            value_org = value
            if row['one_hot_encoded']:
                variable = '_'.join(row['variable'].split('_')[i] for i in range(len(row['variable'].split('_')) - 1))
                value = row['variable'].split('_')[-1]
            expected_value = row['expected_value']
            total_count = len(df[df[variable] == value])
            cluster_count = len(cluster_data[cluster_data[variable] == value])
            expected_proportion = (expected_value / total_count) * 100 if total_count > 0 else 0
            proportion = (cluster_count / total_count) * 100 if total_count > 0 else 0
            percentage_diff = ((proportion - expected_proportion) / expected_proportion) * 100 if expected_proportion != 0 else 0
            
            # Size of the circle proportional to the length of the asterisk
            size = row['log_pvalue']/30
            
            # Draw the triangle
            x = clusters.index(cluster)
            y = list(significant_vars).index(variable_org)
            if value_org == 0:
                # Upward triangle
                triangle = plt.Polygon([(x, y - size), (x - size, y + size), (x + size, y + size)], color=cluster_colors[cluster], alpha=1)
            else:
                # Downward triangle
                triangle = plt.Polygon([(x, y + size), (x - size, y - size), (x + size, y - size)], color=cluster_colors[cluster], alpha=1)
            ax.add_patch(triangle)
        
            # Write odds ratio
            odds_ratio = row['odds_ratio']
            ax.text(x, y + size + 0.1, f'{odds_ratio:.2f}', ha='left', va='center', fontsize=20)


    # Set the labels for the axes
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, fontsize=30)
    ax.set_yticks(range(len(significant_vars)))
    ax.set_yticklabels(significant_names, ha='right', fontsize=30)

    # Adjust the aspect ratio and limits
    ax.set_aspect('equal')

    plt.xlabel('Clusters', fontsize=30)
    plt.title('Significant associations between clusters and discrete features', fontsize=50)
    plt.tight_layout()
    # plt.savefig(fr"/home/pps21@isd.csc.mrc.ac.uk/pps21/paper/motion-signatures/figures/discrete_var.pdf", format='pdf', dpi=300)
    plt.show()