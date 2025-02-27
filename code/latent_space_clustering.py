# Latent signatures exploration

# Change directory to the project code directory
%cd /home/pps21@isd.csc.mrc.ac.uk/pps21/projects/temporal-traj/code

# Import necessary libraries
import random
import sys
import os

import warnings

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

from tableone import TableOne

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchview import draw_graph

# Append custom tools to system path
sys.path.append(r"/home/pps21@isd.csc.mrc.ac.uk/pps21/projects/mytools")
from mesh_utils import normalise_traj2d
from stats_utils import perform_pca, plot_corr_heatmap
from plot_utils import plot_trajectories, plot_settings, plot_interactions, plot_morph

# Settings
pd.options.display.float_format = lambda x: f"{x:.2e}" if abs(x) < 0.001 else f"{x:.4f}"
matplotlib.style.use('ggplot')
warnings.simplefilter('once')
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.multiprocessing.set_sharing_strategy('file_system')
plot_settings()

def determine_optimal_clusters(X, n_used, max_clusters=10):
    """
    Determine the optimal number of clusters using silhouette scores.

    Args:
        X (np.ndarray): The data to cluster.
        n_used (int): Number of principal components used.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        int: Optimal number of clusters based on silhouette scores.
    """   
    # Silhouette Score
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", 
                                max_iter=50, random_state=1, n_jobs=num_workers)
        cluster_labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X.reshape(-1, int(np.ceil(N_FRAME/stride)), n_used), 
                                          cluster_labels, metric="dtw")
        silhouette_scores.append((n_clusters, silhouette_avg))
    
    plt.figure(figsize=(10, 5))
    plt.plot([score[0] for score in silhouette_scores], [score[1] for score in silhouette_scores], 
             marker='o')
    plt.title('Silhouette Scores For Optimal n_clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Return the optimal number of clusters based on the highest silhouette score
    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_clusters

def ts_kmeans(df, n_used, n_clusters, silhouette=False):
    """
    Perform TimeSeries KMeans clustering on the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        n_used (int): Number of principal components used.
        n_clusters (int): Number of clusters to form.
        silhouette (bool): Whether to calculate silhouette score.

    Returns:
        tuple: DataFrame with cluster IDs, the clustering model, and optionally the silhouette score.
    """
    patient_trajectories = df[[f'pc_{x}' for x in range(1,n_used+1)]]
    X = patient_trajectories.to_numpy().reshape(-1, int(np.ceil(N_FRAME/(stride*skip))), n_used)

    optimal_clusters = n_clusters
    if not n_clusters:
        optimal_clusters = determine_optimal_clusters(X, n_used, max_clusters=10)
    model = TimeSeriesKMeans(n_clusters=optimal_clusters, metric="dtw",
                            max_iter=50, random_state=1, n_jobs=num_workers)
    cluster_labels = model.fit_predict(X)

    # Add the cluster IDs as a new column in original dataframe
    patient_trajectories['cluster_id'] = np.repeat(cluster_labels, 
                                                   int(np.ceil(N_FRAME/(stride*skip))))
    df_cluster = patient_trajectories

    if silhouette:
        silhouette_avg = silhouette_score(X.reshape(-1, int(np.ceil(N_FRAME/stride)), n_used), 
                                          cluster_labels, metric="dtw", n_jobs=num_workers)
        return df_cluster, model, silhouette_avg
    else:
        return df_cluster, model

def measure_cluster_stability(original_df_cluster, df, n_used, n_clusters, n_splits=10):
    """
    Measure the stability of clusters using Adjusted Rand Index (ARI).

    Args:
        original_df_cluster (pd.DataFrame): Original DataFrame with cluster IDs.
        df (pd.DataFrame): DataFrame containing the data.
        n_used (int): Number of principal components used.
        n_clusters (int): Number of clusters to form.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        list: List of ARI scores for each split.
    """
    ari_scores = []

    # Extract unique values from the first level of the index
    unique_ids = df.index.get_level_values(0).unique()

    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(unique_ids):
        # Use the test_index to select the corresponding rows from the DataFrame
        subset_ids = unique_ids[test_index]
        df_subset = df.loc[subset_ids]

        # Perform clustering on the subset
        subset_df_cluster, model = ts_kmeans(df_subset, n_used, n_clusters)
        subset_labels = subset_df_cluster.reset_index().drop_duplicates('patient_id')['cluster_id'].values

        # Calculate ARI between the original labels and the subset labels
        subset_original_labels = original_df_cluster.loc[subset_ids, 'cluster_id'].reset_index().drop_duplicates('patient_id')['cluster_id'].values
        ari = adjusted_rand_score(subset_original_labels, subset_labels)
        ari_scores.append(ari)

    return ari_scores

def transform_fields(df, df_mapping):
    """
    Transform fields in the DataFrame based on a mapping.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        df_mapping (pd.DataFrame): DataFrame containing the field mapping.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    list_vars = ['demographics', 'body_measurement', 'risk_factor', 
    'symptoms', 'diagnosis', 'clinical_measurements',
    'genetics', 'biomarkers', 'lifestyle', 'mri_measurements']
    remove_field = ['qualifications', 'employment_status', 'tv_time', 'computer_time',
                    'driving_time', 'veg_intake', 'tea_intake', 'coffee_intake',
                    'fvc', 'fev1', 'pef', 'hdl_cholesterol']
    df['Ell_Global'] = df['Ell_Global'].abs()
    df['Ecc_Global'] = df['Ecc_Global'].abs()
    df['Err_Global'] = df['Err_Global'].abs()
    df = df.drop(columns=remove_field)
    df = df.loc[:,df.columns.isin(df_mapping[df_mapping.category.isin(list_vars)]['field_var'].values)]
    return df

def main():
    """
    Main function to execute the entire process of loading data, performing clustering,
    and evaluating the results.
    """
    # Set seeds for reproducibility
    SEED = 0
    random.seed(SEED)          # Python's random module
    np.random.seed(SEED)       # NumPy's random module
    torch.manual_seed(SEED)    # PyTorch's CPU random module
    torch.cuda.manual_seed(SEED)  # PyTorch's CUDA random module

    ## Parameters and paths ##
    # User-defined parameters
    pca_n = 10

    # Data location
    CLIN_DIR = r"/home/pps21@isd.csc.mrc.ac.uk/pps21/data/ukb_master_clinical.csv"
    UKBB_FIELD_DIR = r"/home/pps21@isd.csc.mrc.ac.uk/pps21/data/ukbb_pheno/ukbb_field_mapping.csv"
    RESULTS_DIR = r'../results'

    ## Import clinical data ##
    df_clin = pd.read_csv(CLIN_DIR)
    df_clin = df_clin.drop(columns=['eid_40616','eid_47602'])
    df_clin = df_clin.sort_values(by='eid_18545').rename(columns={'eid_18545':'patient_id'}).set_index('patient_id')
    # Remove all local phenotypes
    df_clin = df_clin.loc[:,~df_clin.columns.str.contains('WT_AHA')]
    df_clin = df_clin.loc[:, ~df_clin.columns.str.contains(r'Ell(?!_Global)', regex=True)]
    df_clin = df_clin.loc[:, ~df_clin.columns.str.contains(r'Ecc(?!_Global)', regex=True)]
    df_clin = df_clin.loc[:, ~df_clin.columns.str.contains(r'Err(?!_Global)', regex=True)]

    df_mapping = pd.read_csv(UKBB_FIELD_DIR)
    df_mapping.field = df_mapping.field.str.split('|').str[0]
    
    ## Import latent space results ##
    df_latent = pd.read_csv('../results/latent_space.csv', index_col=[0,1])
    # df_latent = pd.read_csv('../results/latent_space_cycle4dnet.csv', index_col=[0,1])
    # df_latent = pd.read_csv('../results/benchmark_model_latent_space.csv', index_col=[0,1]) # if benchmark model

    # Standardisation and plot heatmap
    df_latent = (df_latent-df_latent.mean())/df_latent.std()
    plot_corr_heatmap(df_latent)

    ## Dimensionality reduction
    df_pca = perform_pca(df_latent, pca_n, plot=True)
    df_pca_geo = normalise_traj2d(df_pca, ['pc_1', 'pc_2', 'pc_3', 'pc_4'], normalisation=False)

    # Normalise latent space
    df_pca = (df_pca-df_pca.mean())/df_pca.std()
    # df_pca = df_pca.groupby('patient_id').mean()

    # Figure for paper: trajectory duality
    index_plot = np.random.choice(df_pca.index.get_level_values('patient_id').unique(), 
                                  100, replace=False)
    df_plot = df_pca.loc[df_pca.index.get_level_values('patient_id').isin(index_plot)]
    df_plot_geo = df_pca_geo.loc[df_pca_geo.index.get_level_values('patient_id').isin(index_plot)]
    plot_trajectories(df_plot, 'all', ['pc_1', 'pc_2'], plot_mean=False)
    plot_trajectories(df_plot_geo, 'all', ['pc_1', 'pc_2'], plot_mean=False)

    # Figure for paper: plot trajectories by quintile
    df_pca_clin = pd.merge(df_clin, df_pca, left_index=True, right_index=True, how='inner')
    list_decile = ['age_mri', 'body_surface_area', 'diastolic_bp', 
                   'systolic_bp', 'LVEDV', 'LVEF', 'LVM', 'WT_Global', 
                   'prs_dcm', 'prs_hcm', 'sex', 'Err_Global']
    for i, var_plot in enumerate(list_decile):
        plot_trajectories(df_pca_clin, 'all', ['pc_1', 'pc_2'], 
                          plot_mean=True, color_var=var_plot)
        
    # Figure paper: plot morphing trajectories
    df_pca_clin.rename_axis(index={'frame': 'snap'}, inplace=True)
    df_pca_clin = pd.merge(df_clin, df_pca, left_index=True, right_index=True, how='inner')
    list_morph = ['age_mri', 'systolic_bp', 'LVEF','WT_Global', 'Ecc_Global',
                    'prs_dcm', 'prs_hcm','hba1c','hdl_cholesterol']
    for morph in list_morph:
        plot_morph(df_pca_clin, 'all', ['pc_1', 'pc_2'], [morph], spaced=False)

    # Morphing discrete
    list_risk = ['essential_hypertension', 'diabetes_diagnosed', 'obesity']
    list_mace = ['stroke', 'cardiac_arrest', 'mi', 'heart_failure']
    list_cm = ['hypertrophic_cardiomyopathy', 'dilated_cardiomyopathy']
    list_comb = list_mace + list_cm
    plot_traj_multi(df_pca_clin, 'all', ['pc_1', 'pc_2'], plot_mean=True, color_vars=list_risk)
    plot_traj_multi(df_pca_clin, 'all', ['pc_1', 'pc_2'], plot_mean=True, color_vars=list_comb)
    
    # Figure paper: plot trajectories by quintile interactions
    df_pca_clin = pd.merge(df_clin, df_pca, left_index=True, right_index=True, how='inner')
    list_decile = ['age_mri', 'body_surface_area', 'diastolic_bp', 
                   'systolic_bp', 'LVEDV', 'LVEF', 'LVM', 'WT_Global', 
                   'prs_dcm', 'prs_hcm', 'sex', 'Err_Global']
    plot_interactions(df_pca_clin, 'all', ['pc_1', 'pc_2'], color_vars=['WT_Global', 'systolic_bp'])
    plot_interactions(df_pca_clin, 'all', ['pc_1', 'pc_2'], color_vars=['LVEF', 'age_mri'])
    plot_interactions(df_pca_clin, 'all', ['pc_1', 'pc_2'], color_vars=[ 'LVEF', 'diastolic_bp'])

    df_pca_clin['same'] = 0
    plot_trajectories(df_pca_clin, 'all', ['pc_1', 'pc_2'], plot_mean=True, color_var='same')

    ## Time series clustering
    # Stability analysis
    n_used = 3
    n_splits = 5
    n_clusters_range = [2,3,4,5]
    dict_df = {'placement': df_pca, 'geometry': df_pca_geo}
    dict_result = {}
    for key, df in dict_df.items():
        df_ari = pd.DataFrame()
        for n_clusters in n_clusters_range:
            original_df_cluster, model = ts_kmeans(df, n_used, n_clusters)
            ari_scores = measure_cluster_stability(original_df_cluster, df, n_used, 
                                                   n_clusters, n_splits=n_splits)
            df_ari[f'{n_clusters}'] = ari_scores
            print(f'Average ARI for {key} and {n_clusters}: {np.mean(ari_scores):.2f}')
        dict_result[key] = df_ari

    # Optimal clusters are the one with the highest average ARI
    optimal_clusters = int(dict_result['placement'].mean().idxmax())
    optimal_clusters_geo = int(dict_result['geometry'].mean().idxmax())

    # Save results of stability analysis
    dict_result['placement'].to_csv(f'../results/ari_placement_n{n_used}_k{n_splits}.csv')
    dict_result['geometry'].to_csv(f'../results/ari_geometry_n{n_used}_k{n_splits}.csv')
    
    # Clustering on spatial placement
    df_cluster, model = ts_kmeans(df_pca, n_used=3, n_clusters=optimal_clusters)
    df_cluster = pd.merge(df_clin, df_cluster, left_index=True, right_index=True, how='inner')

    # Clustering on trajectory geometry
    df_cluster_geo, model = ts_kmeans(df_pca_geo, n_used=3, n_clusters=optimal_clusters_geo)
    df_cluster_geo = pd.merge(df_clin, df_cluster_geo, left_index=True, right_index=True, how='inner')

    # Create new combined cluster ID
    df_cluster_mix = pd.merge(df_cluster, df_cluster_geo[['cluster_id']], 
                              left_index=True, right_index=True, suffixes=('', '_geo'))
    df_cluster_mix['cluster_mix'] = df_cluster_mix['cluster_id'].astype(str) + 'x' + df_cluster_mix['cluster_id_geo'].astype(str)

    # Plot average trajectories
    plot_trajectories(df_cluster_mix, 'all', ['pc_1', 'pc_2'], plot_mean=True, color_var='cluster_mix')

    # Save cluster ID for 4D model dynamism analysis
    df_cluster_mix_save = df_cluster_mix.reset_index().drop_duplicates('patient_id')[['patient_id', 
                                                                                 'cluster_mix', 
                                                                                 'cluster_id', 
                                                                                 'cluster_id_geo']]
    df_cluster_mix_save.to_csv('../results/cluster_id_mix.csv')
    # df_cluster_mix_save.to_csv('../results/benchmark_model_cluster_id_mix.csv') # if benchmark model

if __name__ == '__main__':
    main()
