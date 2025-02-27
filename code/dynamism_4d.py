'''
This script processes 4D VTK files to compute velocity and gradient cycles of 3D meshes over time.
It includes functions to:
1. Compute the velocity of mesh points between consecutive VTK files.
2. Compute the gradient cycle of mesh points over a series of VTK files.

'''

import numpy as np
import os
import glob
import numpy as np
import pyvista as pv

# %matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') #png
pv.global_theme.allow_empty_mesh = True

def compute_velocity(vtk_files, global_norm=True):
    """
    Compute the velocity of mesh points between consecutive VTK files.

    Args:
        vtk_files (list): List of VTK file paths.
        global_norm (bool): Whether to normalize velocities globally.

    Returns:
        list: List of normalized velocities for each time step.
    """
    velocities = []
    all_velocities = []

    # Compute velocities and collect all velocity values
    for i in range(1, len(vtk_files)):
        mesh_prev = pv.read(vtk_files[i - 1])
        mesh_curr = pv.read(vtk_files[i])
        mesh_prev = mesh_prev.decimate(0.9)
        mesh_curr = mesh_curr.decimate(0.9)
        velocity = np.sqrt(np.sum((mesh_curr.points - mesh_prev.points) ** 2, axis=1))
        velocities.append(velocity)
        all_velocities.extend(velocity)

    if not global_norm:
        return velocities
    
    # Convert all velocities to a numpy array for easy min/max computation
    all_velocities = np.array(all_velocities)
    global_min = all_velocities.min()
    global_max = all_velocities.max()

    # Normalize velocities using the global min and max
    normalised_velocities = [(velocity - global_min) / (global_max - global_min) for velocity in velocities]

    return normalised_velocities

def compute_gradient_cycle(vtk_files):
    """
    Compute the gradient cycle of mesh points over a series of VTK files.

    Args:
        vtk_files (list): List of VTK file paths.

    Returns:
        np.ndarray: Gradient of the mesh points over time.
    """
    meshes = [pv.read(vtk_file) for vtk_file in vtk_files]
    # meshes = [mesh.decimate(0.9) for mesh in meshes]
    meshes = [mesh.points for mesh in meshes]
    meshes = np.stack(meshes, axis=0)

    # Calculate the bounding box volume
    min_coords = np.min(meshes, axis=(0, 1))
    max_coords = np.max(meshes, axis=(0, 1))
    bounding_box_volume = np.prod(max_coords - min_coords)/(10**6)
        
    # # Compute the root-mean-square distance of points from the centroid
    # centroid = np.mean(meshes, axis=(0, 1))
    # rms_distance = np.sqrt(np.mean(np.sum((meshes - centroid) ** 2, axis=2)))
    gradient = np.gradient(meshes, axis=0) / bounding_box_volume

    return gradient

def compute_differences(velocities, velocities_ref):
    """
    Compute the differences between two sets of velocities.

    Args:
        velocities (list): List of velocities.
        velocities_ref (list): List of reference velocities.

    Returns:
        list: List of differences between velocities and reference velocities.
    """
    differences = []
    for vel, vel_ref in zip(velocities, velocities_ref):
        difference = vel - vel_ref
        differences.append(difference)
    return differences

def extract_integer(filename, position_from_end=1):
    """
    Extract an integer from a filename based on its position from the end.

    Args:
        filename (str): The filename to extract the integer from.
        position_from_end (int): The position of the integer from the end of the filename.

    Returns:
        int: The extracted integer, or -1 if no integer is found.
    """
    filename = filename.replace('.vtk', '')
    parts = filename.split('_')
    matches = [part for part in parts if part.isdigit()]
    if matches and position_from_end <= len(matches):
        return int(matches[-position_from_end])
    return -1

def plot_motion_diff(vtk_files, gradients, clim, mode, anot, view, skip=10, bounds='', gradient_mean=None):
    """
    Plot the motion differences of mesh points over time.

    Args:
        vtk_files (list): List of VTK file paths.
        gradients (np.ndarray): Gradients of the mesh points over time.
        clim (tuple): Color limits for the plot.
        mode (str): Mode of the plot ('diff' or 'absolute').
        anot (str): Annotation for the plot.
        view (str): View angle for the plot.
        skip (int): Number of frames to skip between plots.
        bounds (str): Bounds for the plot.
        gradient_mean (np.ndarray): Mean gradient for comparison.
    """
    mag = 70 # Magnitude of the gradient arrows - adjust as needed
    horizontal = True # Set to True for horizontal layout

    # Plot settings
    num_plots = gradients.shape[0]
    num_subplots = (num_plots + skip - 1) // skip
    plot_shape = (num_subplots, 1)
    window_size=(int(1240/2), int(1240*2)) # (width, height)
    if horizontal:
        plot_shape = plot_shape[::-1]
        window_size = window_size[::-1]

    pv.set_jupyter_backend('static')
    plotter = pv.Plotter(shape=plot_shape, notebook=True, border=False, window_size=window_size, 
                         lighting='none'
                         )
    plotter.reset_camera()

    for idx, i in enumerate(range(0, num_plots, skip)):
        if horizontal:
            plotter.subplot(0, idx)
        else:
            plotter.subplot(idx, 0)

        # Get the mesh and compute normals (the original ones are wrong)
        mesh = pv.read(vtk_files[i])
        mesh.point_data['original_indices'] = np.arange(mesh.n_points)
        mesh = mesh.decimate_pro(0.9, preserve_topology=True)
        decimated_indices = mesh.point_data['original_indices']
        points = mesh.points
        mesh.compute_normals(inplace=True)
        normals = mesh.point_normals

        # Get the gradient vectors and magnitudes
        gradient = np.mean(gradients[i:i + skip], axis=0) # average gradient over skip frames
        gradient = gradient[decimated_indices]
        vectors = gradient.reshape(-1, 3)
        magnitudes = np.linalg.norm(vectors, axis=1)

        # Color surface with gradient magnitude coloured by vector product with points (// center of shape)
        gradient_surface = np.zeros(mesh.n_points)
        all_positive_mask = np.einsum('ij,ij->i', vectors, points) >= 0
        all_negative_mask = np.einsum('ij,ij->i', vectors, points) < 0
        gradient_surface[all_positive_mask] = magnitudes[all_positive_mask]
        gradient_surface[all_negative_mask] = -magnitudes[all_negative_mask]

        # Plot the mesh with gradient magnitude
        cmap = 'RdBu'
        plotter.add_mesh(mesh, scalars=gradient_surface, cmap=cmap, clim=clim, show_edges=False, 
                         smooth_shading=True, show_scalar_bar=False)
        
        ## Plot gradient difference beams
        if mode == 'diff':
            # Compute the gradient difference
            gradient_diff = np.mean(gradients[i:i + skip], axis=0) - np.mean(gradient_mean[i:i + skip], axis=0)
            gradient_diff = gradient_diff[decimated_indices]
            vectors_diff = gradient_diff.reshape(-1, 3)
            magnitudes_diff = np.linalg.norm(vectors_diff, axis=1)

            threshold = 0
            top_mask = magnitudes_diff >= threshold
            top_points = points[top_mask]
            top_vectors = vectors_diff[top_mask]
            top_normals = points[top_mask]

            # Combination 1: positive_mask & all_negative_mask
            positive_mask = np.einsum('ij,ij->i', top_vectors, top_normals) > 0
            mask2 = positive_mask & all_negative_mask
            points2 = top_points[mask2]
            vectors2 = top_vectors[mask2]
            coord2 = np.array([coord for pair in zip(points2, points2 + mag * vectors2) for coord in pair])
            if len(coord2)>0:
                plotter.add_lines(coord2, color='#0100CE', width=1)  # Blue beams on red surface

            # Combination 2: negative_mask & all_positive_mask
            negative_mask = np.einsum('ij,ij->i', top_vectors, top_normals) < 0
            mask3 = negative_mask & all_positive_mask
            points3 = top_points[mask3]
            vectors3 = top_vectors[mask3]
            points3 = points3 - mag * vectors3
            coord3 = np.array([coord for pair in zip(points3, points3 + mag * vectors3) for coord in pair])  
            if len(coord3)>0:
                plotter.add_lines(coord3, color='#CE0100', width=1) # Red beams on blue surface

        plotter.camera.azimuth = 0
        plotter.set_position(VIEWS[view])

    # plotter.save_graphic(filename=f"results/motion_{anot}_{mode}_gradient_{view}.pdf", raster=False)
    plotter.show()

def main(): 
    """
    Main function to execute the entire process of loading VTK files, computing gradients,
    and plotting the motion differences.
    """
    # Directory and parameters
    MEAN_DIR = r"P:/data/decim/motion_0.97/mean_mesh"
    CLUSTER_DIR = r"P:/projects/temporal-traj/results/mesh_cluster"
    VIEWS = {'anterior_lateral': [160,160,160],
             'posterior_septal':  [-160,-160,160],}
    clusters = ['2x1', '2x0', '1x1', '0x1', '0x0', '1x0']

    # Get all vtk files in the directories
    vtk_mean_files = glob.glob(os.path.join(MEAN_DIR, '*.vtk'))
    vtk_files = {cluster: glob.glob(os.path.join(CLUSTER_DIR, cluster, '*.vtk')) for cluster in clusters}

    # Temporary fix of disalignment
    temp_index = list(range(50))[0:48]
    
    # Rank vtk_files based on the integer at the specified position from the end of the filename
    vtk_mean_files = [sorted(vtk_mean_files, key=lambda x: extract_integer(x, position_from_end=1))[i] for i in temp_index]
    vtk_files = {cluster: [sorted(vtk_files[cluster], key=lambda x: extract_integer(x, position_from_end=1))[i] for i in temp_index] for cluster in clusters}

    # Compute cardiac cycle gradient
    gradient_mean = compute_gradient_cycle(vtk_mean_files)
    gradients = {cluster: compute_gradient_cycle(vtk_files[cluster]) for cluster in clusters}

    # Temporary fix of disalignment
    right_index = list(range(50))[0:19] + list(range(50))[23:48]
    vtk_files = {key: [dic[i] for i in right_index] for key, dic in vtk_files.items()}
    vtk_mean_files = [vtk_mean_files[i] for i in right_index]
    gradient_mean = gradient_mean[right_index]
    gradients = {key: gradients[key][right_index] for key in clusters}

    # Bounds and views
    max_gradient = np.max([np.max(np.linalg.norm(grad, axis=2)) for grad in list(gradients.values())])
    clim = [-max_gradient/2, max_gradient/2]
    bounds = list(max(mesh.bounds for mesh in pv.read(vtk_mean_files)))
    view_choice = ['anterior_lateral', 'posterior_septal']
    
    # Plot 4D differences of velocity
    [plot_motion_diff(vtk_files[cluster], gradients[cluster], 
                clim, mode='diff', anot=cluster, 
                view=view, skip=8, 
                bounds=list(bounds),
                gradient_mean=gradient_mean) for cluster in clusters[:] for view in view_choice[:]]
    
    # Plot population average velocity
    [plot_motion_diff(vtk_mean_files, gradient_mean, 
                    clim, mode='absolute', anot='mean', 
                    view=view, skip=8, 
                    gradient_mean=gradient_mean) for view in view_choice[:]]