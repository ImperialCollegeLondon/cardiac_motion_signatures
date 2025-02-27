import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

to_pil_image = transforms.ToPILImage()

def loader_to_numpy(loader):
    """
    Converts a PyTorch DataLoader to a NumPy array.
    
    Parameters:
    loader (DataLoader): The DataLoader containing the data.
    
    Returns:
    np.ndarray: The data as a NumPy array.
    """
    array = []
    for data in loader:
        array.append(data[0].cpu().numpy())
    array = np.vstack(array)
    return array

def create_data_loaders(data, shuffle, batch_size=64, resize_shape=0, resize_dim=0, num_workers=1):
    """
    Creates PyTorch DataLoader objects from the given data.
    
    Parameters:
    data (np.ndarray): The data to be loaded.
    shuffle (bool): Whether to shuffle the data.
    batch_size (int): The batch size for the DataLoader.
    resize_shape (int): The shape to resize the data to (if applicable).
    resize_dim (int): The dimension to resize (if applicable).
    num_workers (int): The number of worker threads to use for loading data.
    
    Returns:
    tuple: A tuple containing the DataLoader and the length of the dataset.
    """
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Resize tensors to a multiple of 2 for the Conv2d layers
    # +1 on resize_dim as tensor add an extra dimension for the batch size
    if resize_shape:
        data_tensor = F.interpolate(data_tensor, size=(resize_shape, data.shape[resize_dim+1]), mode='bilinear', align_corners=False)

    # Create TensorDataset objects
    dataset = TensorDataset(data_tensor)

    # Create DataLoader objects
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader, len(dataset)

def show_loss_plot(train_loss, valid_loss):
    """
    Plots the training and validation loss over epochs.
    
    Parameters:
    train_loss (list): List of training loss values.
    valid_loss (list): List of validation loss values.
    
    Returns:
    None
    """
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../outputs/loss.jpg')
    plt.show()

def beta_kl_rec_loss(mse_loss, mu, logvar, beta=1):
    """
    Computes the combined reconstruction loss (MSE) and KL-Divergence loss.
    
    Parameters:
    mse_loss (torch.Tensor): The reconstruction loss.
    mu (torch.Tensor): The mean from the latent vector.
    logvar (torch.Tensor): The log variance from the latent vector.
    beta (float): The weight of the KL-Divergence term.
    
    Returns:
    tuple: A tuple containing the MSE loss, KL-Divergence loss, and the total loss.
    """
    MSE = mse_loss.sum()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = beta*KLD
    return MSE, KLD, MSE + KLD

def print_losses(total_loss, mse_loss, kld_loss):
    """
    Prints the MSE, KLD, and total losses for the current epoch.
    
    Parameters:
    total_loss (float): The total loss.
    mse_loss (float): The MSE loss.
    kld_loss (float): The KL-Divergence loss.
    
    Returns:
    None
    """
    print(f"Total Loss: {total_loss:.4f}, MSE Loss: {mse_loss:.4f}, KLD Loss: {kld_loss:.4f}")

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) class.
    """
    def __init__(self, kernel_size=4, init_channels=8, image_channels=3, latent_dim=16,
                 padding=0):
        """
        Initializes the ConvVAE model.
        
        Parameters:
        kernel_size (int): The size of the convolutional kernels.
        init_channels (int): The number of initial channels.
        image_channels (int): The number of channels in the input images.
        latent_dim (int): The dimensionality of the latent space.
        padding (int): The padding for the convolutional layers.
        """
        super(ConvVAE, self).__init__()
        
        self.kernel_size = kernel_size
        self.init_channels = init_channels
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.padding = padding

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(init_channels*4*2, init_channels*8*2)
        self.fc_mu = nn.Linear(init_channels*8*2, latent_dim)
        self.fc_log_var = nn.Linear(init_channels*8*2, latent_dim)
        self.fc2 = nn.Linear(latent_dim, init_channels*4*2)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=init_channels*4*2, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=padding
        )
    def reparameterize(self, mu, log_var):
        """
        Reparameterizes the latent space using the mean and log variance.
        
        Parameters:
        mu (torch.Tensor): The mean from the encoder's latent space.
        log_var (torch.Tensor): The log variance from the encoder's latent space.
        
        Returns:
        torch.Tensor: The reparameterized latent vector.
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        tuple: A tuple containing the reconstruction, mean, log variance, and latent vector.
        """
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 1, self.init_channels*4*2, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)
        reconstruction = x
        return reconstruction, mu, log_var, z

    def train_model(self, dataloader, dataset_len, device, optimizer, criterion, beta):
        """
        Trains the VAE model.
        
        Parameters:
        dataloader (DataLoader): The DataLoader for the training data.
        dataset_len (int): The length of the dataset.
        device (torch.device): The device to train on (CPU or GPU).
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.Module): The loss function.
        beta (float): The weight of the KL-Divergence term.
        
        Returns:
        float: The average training loss.
        """
        self.train()
        running_loss = 0.0
        running_mse = 0.0
        running_kld = 0.0
        counter = 0
        for i, data in tqdm(enumerate(dataloader), total=int(dataset_len/dataloader.batch_size)):
            counter += 1
            data = data[0]
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar, z_latent_space = self(data)
            mse_loss = criterion(reconstruction, data)
            mse, kld, loss = beta_kl_rec_loss(mse_loss, mu, logvar, beta)
            loss.backward()
            running_loss += loss.item()
            running_mse += mse.item()
            running_kld += kld.item()
            optimizer.step()
        train_loss = running_loss / counter 
        mse_loss = running_mse / counter
        kld_loss = running_kld / counter

        print_losses(train_loss, mse_loss, kld_loss)
        return train_loss

    def validate(self, dataloader, dataset_len, device, criterion, beta):
        """
        Validates the VAE model.
        
        Parameters:
        dataloader (DataLoader): The DataLoader for the validation data.
        dataset_len (int): The length of the dataset.
        device (torch.device): The device to validate on (CPU or GPU).
        criterion (torch.nn.Module): The loss function.
        beta (float): The weight of the KL-Divergence term.
        
        Returns:
        tuple: A tuple containing the average validation loss and the reconstructed images.
        """
        self.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_kld = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(dataset_len/dataloader.batch_size)):
                counter += 1
                data= data[0]
                data = data.to(device)
                reconstruction, mu, logvar, z_latent_space = self(data)
                mse_loss = criterion(reconstruction, data)
                mse, kld, loss = beta_kl_rec_loss(mse_loss, mu, logvar, beta)
                running_loss += loss.item()
                running_mse += mse.item()
                running_kld += kld.item()
            
                # save the last batch input and output of every epoch
                if i == int(dataset_len/dataloader.batch_size) - 1:
                    recon_images = reconstruction
        val_loss = running_loss / counter
        mse_loss = running_mse / counter
        kld_loss = running_kld / counter

        print_losses(val_loss, mse_loss, kld_loss)
        return val_loss, recon_images

    def predict(self, dataloader, dataset_len, device):
        """
        Generates predictions using the VAE model.
        
        Parameters:
        dataloader (DataLoader): The DataLoader for the data.
        dataset_len (int): The length of the dataset.
        device (torch.device): The device to predict on (CPU or GPU).
        
        Returns:
        tuple: A tuple containing the reconstructed images, mean, log variance, and latent vectors.
        """
        self.eval()
        z_latent_batch = []
        mu_batch = []
        reconstruction_batch = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(dataset_len/dataloader.batch_size)):
                data= data[0]
                data = data.to(device)
                reconstruction, mu, logvar, z_latent_space = self(data)

                z_latent_batch.append(z_latent_space)
                mu_batch.append(mu)
                reconstruction_batch.append(reconstruction)

        return reconstruction_batch, mu_batch, logvar, z_latent_batch
    