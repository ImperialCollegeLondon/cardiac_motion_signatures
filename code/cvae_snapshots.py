# Convolutional variational autoencoder for motion snapshots data

# Change directory to the project code directory
%cd /home/pps21@isd.csc.mrc.ac.uk/pps21/projects/temporal-traj/code

# Import necessary libraries
import random
import sys
import os

import warnings
import gc
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchview import draw_graph

# Append custom tools to system path
sys.path.append(r"/home/pps21@isd.csc.mrc.ac.uk/pps21/projects/mytools")
from mesh_utils import process_motion, normalise_cloud, plot_example_3d
from stats_utils import plot_mae, plot_chamfer, perform_pca
from plot_utils import plot_settings
from deepl_utils import ConvVAE, create_data_loaders, show_loss_plot, loader_to_numpy

# Settings
pd.options.display.float_format = lambda x: f"{x:.2e}" if abs(x) < 0.001 else f"{x:.4f}"
matplotlib.style.use('ggplot')
warnings.simplefilter('once')
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.multiprocessing.set_sharing_strategy('file_system')
plot_settings()

def process_batch(batch_data):
    """
    Process a batch of data using parallel processing.

    Args:
        batch_data (pd.DataFrame): The batch of data to process.

    Returns:
        tuple: Processed data and corresponding index.
    """
    with ThreadPoolExecutor(num_workers) as executor:
        future = executor.submit(process_subjects, batch_data)
    return future.result()

def process_subjects(df):
    """
    Process motion data for each subject.

    Args:
        df (pd.DataFrame): DataFrame containing the motion data.

    Returns:
        tuple: Processed data and corresponding index.
    """
    df = process_motion(df, N_FRAME)
    df = df.groupby(level=0).apply(create_snaps, window=window, stride=stride, 
                                    skip=skip, periodicity=True)
    df = arrange_snaps(df)
    df = columns_3d(df)
    df = normalise_cloud(df)
    index = df.index.droplevel(['pos', 'Coordinate']).drop_duplicates()
    df = df.to_numpy().reshape(-1, window, 3, df.shape[1])
    df = df.transpose(0, 2, 3, 1)
    return df, index

def read_chunks(MOTION_DIR, chunk_size):
    """
    Read data in chunks from a CSV file using parallel processing.

    Args:
        MOTION_DIR (str): Path to the motion data CSV file.
        chunk_size (int): Size of each chunk to read.

    Returns:
        list: List of data chunks.
    """
    chunks = []
    with ThreadPoolExecutor(num_workers) as executor:
        for chunk in pd.read_csv(MOTION_DIR, index_col=[0,1], chunksize=chunk_size):
            chunk = executor.submit(lambda x: x, chunk).result()
            chunks.append(chunk)
    return chunks

def create_snaps(df, window, stride=1, skip=1, periodicity=True):
    """
    Create snapshots from the motion data.

    Args:
        df (pd.DataFrame): The motion data.
        window (int): The window size for snapshots.
        stride (int): The stride for creating snapshots.
        skip (int): The number of frames to skip.
        periodicity (bool): Whether to consider periodicity.

    Returns:
        pd.DataFrame: DataFrame containing the snapshots.
    """
    df = df.iloc[::skip] # Remove every skip frames
    snaps = []
    if periodicity:
        df = pd.concat([df, df.iloc[0:window-1]]) # add beginning of cycle for periodicity
    for i in range(0, len(df)-window+1, stride):
        df_temp = df.iloc[i : i + window]
        df_temp['snap'] = i
        snaps.append(df_temp.reset_index(drop=True))
        del df_temp
    return pd.concat(snaps)

def arrange_snaps(df_snap):
    """
    Arrange snapshots in a specific format.

    Args:
        df_snap (pd.DataFrame): DataFrame containing the snapshots.

    Returns:
        pd.DataFrame: Arranged DataFrame.
    """
    df_snap.index = df_snap.index.set_names(['patient_id', 'pos'])
    df_snap = df_snap.reset_index()
    df_snap = df_snap.set_index(['patient_id', 'snap', 'pos'])
    return df_snap

def columns_3d(df_snap):
    """
    Convert columns to a 3D format.

    Args:
        df_snap (pd.DataFrame): DataFrame containing the snapshots.

    Returns:
        pd.DataFrame: DataFrame with 3D columns.
    """
    column_tuples = [(col.split('_')[0], int(col.split('_')[1])) for col in df_snap.columns]
    multi_index = pd.MultiIndex.from_tuples(column_tuples, names=['Coordinate', 'Number'])
    df_snap.columns = multi_index

    df_snap = df_snap.sort_index(axis=1, level='Number')
    df_snap = df_snap.stack(level='Coordinate')
    df_snap = df_snap.sort_index()
    
    return df_snap

def plot_single_example(snaps_polate, reconstruction, patient_index, snap_index, pos_frame):
    """
    Plot a single example of the original and reconstructed snapshots.

    Args:
        snaps_polate (np.ndarray): Interpolated snapshots.
        reconstruction (np.ndarray): Reconstructed snapshots.
        patient_index (int): Index of the patient.
        snap_index (int): Index of the snapshot.
        pos_frame (int): Frame position to plot.
    """
    index = patient_index * int(np.ceil(N_FRAME/(stride*skip))) + snap_index
    matrix1 = snaps_polate[index,:,:,pos_frame].transpose(1,0)
    matrix2 = reconstruction[index,:,:,pos_frame].transpose(1,0)
    plot_example_3d(matrix2)
    plot_example_3d(matrix1)
    plot_example_3d(matrix1, matrix2)

def main():
    """
    Main function to execute the entire process of loading data, training the model,
    and evaluating the results.
    """
    # Set seeds for reproducibility
    SEED = 0
    random.seed(SEED)          # Python's random module
    np.random.seed(SEED)       # NumPy's random module
    torch.manual_seed(SEED)    # PyTorch's CPU random module
    torch.cuda.manual_seed(SEED)  # PyTorch's CUDA random module

    # Optional: Ensure deterministic behavior in PyTorch
    cudnn.benchmark = True
    cudnn.deterministic = False  # Set to True if you want deterministic behavior

    ## Parameters and paths ##
    # User-defined parameters
    window = 16
    stride = 4
    skip = 1
    pca_n = 10

    # Model parameters
    num_workers = 35
    test_split = 0.2
    lr = 0.001
    epochs = 25
    batch_size = 256
    beta = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} of {torch.cuda.get_device_name(0)}')

    # Data location
    MOTION_DIR = r"/home/pps21@isd.csc.mrc.ac.uk/pps21/data/decim/motion_0.97/LVmyo_all_mesh_motion_decim_0.97.csv"
    N_FRAME = 50

    ## Import motion data with parallel processing ##

    chunk_size = 1000
    chunks = read_chunks(MOTION_DIR, chunk_size)
    all_chunks = []
    indexes = []
    for chunk in chunks:
        chunk, index = process_batch(chunk)
        all_chunks.append(chunk)
        indexes.append(index)
        gc.collect()
    all_arrays = np.concatenate(all_chunks, axis=0)
    all_index = np.concatenate(indexes, axis=0)
    del all_chunks, indexes, chunks

    ## Training model
    # Create data loaders
    train_data, test_data = train_test_split(all_arrays[:,:,:,:], test_size=test_split, 
                                             random_state=1)
    trainloader, trainset_len = create_data_loaders(train_data, shuffle=True, batch_size=batch_size,
                                                    resize_shape=1024, resize_dim=2, 
                                                    num_workers=num_workers)
    testloader, testset_len = create_data_loaders(test_data, shuffle=False, batch_size=batch_size,
                                                resize_shape=1024, resize_dim=2, 
                                                num_workers=num_workers)

    # Initialise the model
    model = ConvVAE(kernel_size=4, init_channels=8, latent_dim=16, padding=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    # Draw model graph
    model_graph = draw_graph(model, input_size=(batch_size, 3, 1024, 16), device='meta')
    model_graph.visual_graph

    # Training loop
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        train_epoch_loss = model.train_model(trainloader, trainset_len, device, 
                                             optimizer, criterion, beta)
        valid_epoch_loss, _ = model.validate(testloader, testset_len, device, criterion, beta)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} of {epochs}")
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")

        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

    # Display training loss plot
    show_loss_plot(train_loss, valid_loss)
    loss = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss})
    loss['epoch'] = loss.index
    loss.to_csv('../results/loss_model.csv')

    # Save the trained model
    torch.save(model.state_dict(), '../results/trained_model.pth')

    ## Evaluation of trained model (unsupervised)
    # Load input and interpolate
    reconstruct_size = temp_size//2
    loader, loaderset_len = create_data_loaders(all_arrays[reconstruct_size:,:,:,:], 
                                                shuffle=False, batch_size=batch_size,
                                                resize_shape=1024, resize_dim=2, 
                                                num_workers=num_workers)
    snaps_polate = loader_to_numpy(loader)

    # Reconstruction from model
    reconstruction, _, _, _ = model.predict(loader, loaderset_len, device)
    reconstruction = torch.cat(reconstruction, dim=0)
    reconstruction = reconstruction.cpu().detach().numpy()

    # Plot reconstruction loss
    mae_percent = plot_mae(snaps_polate, reconstruction)
    # pd.DataFrame(mae_PERCENT, columns=['X', 'Y', 'Z']).to_csv('../results/mae_percent_3d_mean.csv')

    chamfer_dist = plot_chamfer(snaps_polate, reconstruction)
    # pd.DataFrame(chamfer_dist).to_csv('../results/chamfer_dist_4-4.csv')

    # Plot single example
    plot_single_example(snaps_polate, reconstruction, patient_index=2, snap_index=2, pos_frame=12)

    ## Latent space exploration
    _, mu, _, _ = model.predict(loader, loaderset_len, device)
    mu = torch.cat(mu, dim=0)
    mu = mu.cpu().detach().numpy()
    df_latent = pd.DataFrame(data=mu, index=pd.MultiIndex.from_tuples(all_index[reconstruct_size:], 
                                                                      names=['patient_id', 'snap']))

    # Save for later or load for restarting analysis
    df_latent.to_csv('../results/latent_space.csv')

    ## Benchmark model
    # PCA directly on motion data
    all_arrays = all_arrays.reshape(all_arrays.shape[0], -1)
    df_arrays = pd.DataFrame(data=all_arrays,
                             index=pd.MultiIndex.from_tuples(all_index,
                                                             names=['patient_id', 'snap']))
    df_pca = perform_pca(df_arrays, pca_n, plot=True)
    df_pca.to_csv('../results/benchmark_model_latent_space.csv')

if __name__ == '__main__':
    main()
