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

# Append custom tools to system path
sys.path.append(r"/home/pps21@isd.csc.mrc.ac.uk/pps21/projects/mytools")
from mesh_utils import normalise_traj2d
from stats_utils import perform_pca, plot_corr_heatmap, \
                        run_test_by_cluster, posthoc_tests, plot_summary_continuous, \
                        plot_grid_by_cluster, plot_zscore, plot_latent_heatmap
from plot_utils import plot_trajectories, plot_settings, plot_cluster_proportions, \
                        plot_trajectories, plot_average_trajectories_subplots, plot_cluster_surfaces

# Settings
pd.options.display.float_format = lambda x: f"{x:.2e}" if abs(x) < 0.001 else f"{x:.4f}"
matplotlib.style.use('ggplot')
warnings.simplefilter('once')
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_settings()

def transform_fields(df, df_mapping):
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
    Main function to execute the entire process of loading data, performing statistical enrichment,
    and evaluating the results.
    """
    # Set seeds for reproducibility
    SEED = 0
    random.seed(SEED)          # Python's random module
    np.random.seed(SEED)       # NumPy's random module

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
    # df_latent = pd.read_csv('../results/benchmark_model_latent_space.csv', index_col=[0,1]) # if benchmark model

    # Standardisation and plot heatmap
    df_latent = (df_latent-df_latent.mean())/df_latent.std()
    plot_corr_heatmap(df_latent)

    ## Dimensionality reduction
    df_pca = perform_pca(df_latent, pca_n, plot=True)
    df_pca_geo = normalise_traj2d(df_pca, ['pc_1', 'pc_2', 'pc_3', 'pc_4'], normalisation=False)

    # Normalise latent space
    df_pca = (df_pca-df_pca.mean())/df_pca.std()

    # Heatmap between latent vectors and subject features
    list_heatmap_cat = ['demographics', 'body_measurement', 
                        'clinical_measurements', 'biomarkers', 'mri_measurements']
    plot_latent_heatmap(df_pca, df_clin[df_clin.columns[df_clin.columns.isin(df_mapping[df_mapping.category.isin(list_heatmap_cat)]['field_var'].values)]], df_mapping)

    # Load cluster ID from previous clustering analysis
    df_cluster_mix = pd.read_csv('../results/cluster_id_mix.csv', index_col=[0])
    # df_cluster_mix = pd.read_csv('../results/benchmark_model_cluster_id_mix.csv', index_col=[0]) # if benchmark model

    df_cluster_mix = df_cluster_mix.set_index('patient_id')
    df_cluster_mix = pd.merge(df_cluster_mix, df_pca, how='inner', left_index=True, right_index=True)
    df_cluster_mix = pd.merge(df_clin, df_cluster_mix, left_index=True, right_index=True, how='inner')

    # Figure paper: average surface, average trajectory and cluster proportions
    plot_cluster_surfaces(df_cluster_mix, ['pc_1', 'pc_2'], 'cluster_mix', alpha=0.1)
    plot_average_trajectories_subplots(df_cluster_mix, ['pc_1', 'pc_2'], 'cluster_mix')
    plot_cluster_proportions(df_cluster_mix, 'cluster_mix')

    # Figure paper: plot trajectories by quintile
    list_decile = ['age_mri', 'body_surface_area', 'diastolic_bp', 
                   'systolic_bp', 'LVEDV', 'LVEF', 'LVM', 'WT_Global', 
                   'prs_dcm', 'prs_hcm', 'sex', 'Err_Global']
    for i, var_plot in enumerate(list_decile):
        plot_trajectories(df_cluster_mix, 'all', ['pc_1_y', 'pc_2_y'], 
                          plot_mean=True, color_var=var_plot)

    ## Statistical findings
    df_test_mix = df_cluster_mix.reset_index().drop_duplicates(subset='patient_id').set_index(['patient_id', 'cluster_mix']).drop('snap', axis=1)
    df_test_mix = df_test_mix.drop(columns=['cluster_id', 'cluster_id_geo'])
    df_test_mix = df_test_mix.loc[:,~df_test_mix.columns.str.contains(r'pc_\d+', regex=True)]

    # Re-order clusters
    list_clusters = ['2x1', '2x0', '1x1', '0x1', '0x0', '1x0']
    df_cluster_mix['cluster_mix'] = pd.Categorical(df_cluster_mix['cluster_mix'], 
                                                   categories=list_clusters, ordered=True)

    df_test_mix = df_test_mix.reset_index()
    df_test_mix['cluster_mix'] = pd.Categorical(df_test_mix['cluster_mix'], 
                                                categories=list_clusters, ordered=True)
    df_test_mix = df_test_mix.sort_values('cluster_mix')
    df_test_mix = df_test_mix.set_index('cluster_mix')

    # Perform all statistical tests
    df_test_mix = transform_fields(df_test_mix, df_mapping)
    df_p_values, df_exp, eta_square, cramerV = run_test_by_cluster(df_test_mix, 'cluster_mix')
    df_dunn, df_hgeom = posthoc_tests(df_test_mix, df_p_values, 'cluster_mix')
    df_hgeom = pd.merge(df_hgeom, df_exp, how='left', 
                        left_on=['cluster_mix', 'variable', 'value'], 
                        right_on=['cluster_mix', 'variable', 'value'])
    
    # Plot results of statistical tests
    plot_summary_continuous(df_test_mix, df_dunn, 'cluster_mix', eta_square, df_mapping)
    plot_grid_by_cluster(df_test_mix, df_hgeom, 'cluster_mix', df_mapping)
    plot_zscore(df_test_mix, df_dunn, 'cluster_mix', df_mapping)

    # Figure for paper
    list_continuous = ['prs_hcm', 'LVM', 'Err_Global', 'Ell_Global', 'LVEDV',
                        'prs_dcm', 'WT_Global', 
                        'systolic_bp', 'body_surface_area', 'age_mri', 'Ecc_Global', 
                        'adjusted_ts_ratio', 'diastolic_bp', 'prs_ht', 'prs_ldl_chol', 'LVEF']
    ci_all = plot_summary_continuous(df_test_mix, df_dunn[df_dunn.variable.isin(list_continuous)], 
                            'cluster_mix', eta_square, df_mapping, z_score=True)
    plot_zscore(df_test_mix, df_dunn[df_dunn.variable.isin(list_continuous)], 
                'cluster_mix', df_mapping)
    
    list_grid = ['wheeze_last_year', 'sob_level_ground', 'ever_smoked', 'afib_flutter',
                 'hypertrophic_cardiomyopathy', 'mitral_valve_disease', 'congenital_hd', 
                 'dyslipidaemia', 'heart_failure',
                 'essential_hypertension', 'dilated_cardiomyopathy', 'diabetes_diagnosed', 
                 'obesity', 'stroke', 'cardiac_arrest',
                 'coronary_atherosclerosis', 'mi', 'sex', 'ethnic_background']
    plot_grid_by_cluster(df_test_mix, df_hgeom[df_hgeom.variable.isin(list_grid)], 
                         'cluster_mix', df_mapping)
    
    # Get values for reporting
    df_report = create_summary_continuous(df_test_mix, df_dunn, list_continuous[0])

if __name__ == '__main__':
    main()
