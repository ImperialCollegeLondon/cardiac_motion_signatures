import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

import sys
sys.path.append(r"/home/pps21@isd.csc.mrc.ac.uk/pps21/projects/mytools")
from stats_utils import run_test_by_cluster, posthoc_tests, plot_summary_continuous, plot_grid_by_cluster

%cd /home/pps21@isd.csc.mrc.ac.uk/pps21/projects/temporal-traj/code

def main():
    # Parameters
    CLIN_DIR = r"/home/pps21@isd.csc.mrc.ac.uk/pps21/data/ukb_master_clinical.csv"
    UKBB_FIELD_DIR = r"/home/pps21@isd.csc.mrc.ac.uk/pps21/data/ukbb_pheno/ukbb_field_mapping.csv"
    RESULTS_DIR = r'../results'

    motion_ids = pd.read_csv('../data/cluster_id_mix.csv', index_col=[0]).patient_id.values

    # Load the dataset
    df_clin = pd.read_csv(CLIN_DIR)
    df_clin = df_clin.drop(columns=['eid_40616','eid_47602'])
    df_clin = df_clin.sort_values(by='eid_18545').rename(columns={'eid_18545':'patient_id'}).set_index('patient_id')
    df_clin = df_clin[df_clin.index.isin(motion_ids)]

    df_mapping = pd.read_csv(UKBB_FIELD_DIR)
    df_mapping.field = df_mapping.field.str.split('|').str[0]

    # Subset on MRI-derived variables
    mri_variables = df_mapping[df_mapping.category=='mri_measurements'].field_var.tolist()
    local_variables = []
    # local_variables = df_clin.loc[:,df_clin.columns.str.contains(r'WT_AHA|Ell|Ecc|Err', regex=True)].columns.tolist()  
    df_mri = df_clin[list(set(mri_variables + local_variables))]
    df_mri = df_mri.dropna()

    # Standardize the data
    scaler = StandardScaler()
    mri_data_scaled = scaler.fit_transform(df_mri)
    # df_mri = pd.DataFrame(mri_data_scaled, columns=df_mri.columns)  

    # Apply UMAP
    umap_reducer = umap.UMAP(n_components=16)
    umap_embedding = umap_reducer.fit_transform(mri_data_scaled)

    # Apply PCA
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(umap_embedding)

    # KFold cross-validation
    kf = KFold(n_splits=5)

    # Loop through n_clusters from 2 to 9
    for n_clusters in range(2, 10):
        # Apply KMeans clustering on the whole dataset
        kmeans = KMeans(n_clusters=n_clusters)
        clusters_whole = kmeans.fit_predict(pca_result)
        
        # Add cluster labels to the original data
        df_mri['cluster_id'] = clusters_whole
        df = pd.merge(df_clin, df_mri[['cluster_id']], left_index=True, right_index=True, how='right')
        
        ari_scores = []
        
        for train_index, test_index in kf.split(mri_data_scaled):
            X_train, X_test = mri_data_scaled[train_index], mri_data_scaled[test_index]
            
            # Apply KMeans clustering
            kmeans_train = KMeans(n_clusters=n_clusters)
            clusters_test = kmeans_train.fit_predict(X_test)
            
            # Calculate ARI score
            ari_score = adjusted_rand_score(clusters_whole[test_index], clusters_test)
            ari_scores.append(ari_score)
        
        # Print ARI scores for the current n_clusters
        print(f"n_clusters = {n_clusters}")
        print("ARI scores for each fold:", ari_scores)
        print("Mean ARI score:", sum(ari_scores) / len(ari_scores))

    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(pca_result)
    # Add cluster labels to the original data
    df_mri['cluster_id'] = clusters
    df = pd.merge(df_clin, df_mri[['cluster_id']], left_index=True, right_index=True, how='right')

    # Test for significant variables with the rest of the clinical variables
    df_test = df[[col for col in df.columns if col not in df_mri.columns] + ['cluster_id']]
    df_test = df_test.reset_index()
    df_test['cluster_id'] = pd.Categorical(df_test['cluster_id'])
    df_test = df_test.sort_values('cluster_id')
    df_test = df_test.set_index('cluster_id')

    # df_test = transform_fields(df_test, df_mapping)
    df_p_values, df_exp, eta_square, cdframerV = run_test_by_cluster(df_test, 'cluster_id')
    df_dunn, df_hgeom = posthoc_tests(df_test, df_p_values, 'cluster_id')

    # Plot results of statistical tests
    df_hgeom = pd.merge(df_hgeom, df_exp, how='left', left_on=['cluster_id', 'variable', 'value'], 
                        right_on=['cluster_id', 'variable', 'value'])
    plot_summary_continuous(df_test, df_dunn, 'cluster_id', eta_square, df_mapping)
    plot_grid_by_cluster(df_test, df_hgeom, 'cluster_id', df_mapping)

if __name__ == '__main__':
    main()
