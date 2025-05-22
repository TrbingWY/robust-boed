import torch
import numpy as np
from scipy.stats import wasserstein_distance, entropy, gaussian_kde
import sys
sys.path.append("./toy_examples/dad_regression/script/fixedmodel")
import os
import pandas as pd
import re


def compute_covarite_shift(points,covarite_shift_method = "wass"):
    if covarite_shift_method == "sum_of_minDistance":
        points = points.view(-1)
        n = points.shape[0]
        pairwise_distances = torch.abs(points.unsqueeze(1) - points.unsqueeze(0))  # (N, N)

        pairwise_distances.fill_diagonal_(float('inf'))

        min_distances = torch.min(pairwise_distances, dim=1)[0]  

        return torch.sum(min_distances)

    if covarite_shift_method == "kde":
        
        points = points.view(-1).detach().cpu().numpy()
        if len(points) < 3:
            return torch.tensor(0.0)
        kde_p = gaussian_kde(points)  
        x_grid = np.linspace(-4, 4, 100)  
        p_x = kde_p(x_grid)  

      
        q_x = np.ones_like(p_x) / (4 - (-4))  # uniform distribution
        p_x += 1e-9
        q_x += 1e-9

        kl_div = entropy(p_x, q_x)

        print("kl div", kl_div)
        return torch.tensor(kl_div, dtype=torch.float32)

    if covarite_shift_method == "wass":
        div = compute_wasserstein_dispersion(points)
        print("wass distance:",div)
        return div
    if covarite_shift_method == "mmd":
        """
        Computes MMD distance between the selected points and a uniform distribution over the same range.
        """
        points = points.view(-1)
        uniform_samples = torch.linspace(-4, 4, 100)  # Generate uniform points in the same range
        div = compute_mmd(points, uniform_samples)
        print("MMD distance:", div)
        return div

def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between two sets of points.
    """
    x = x.view(-1, 1)
    y = y.view(-1, 1)

    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    xy = torch.mm(x, y.T)

    pairwise_distances = xx - 2 * xy + yy.T
    return torch.exp(-pairwise_distances / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) distance between two distributions x and y.
    Uses an RBF kernel.
    """
    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)

    m = x.shape[0]
    n = y.shape[0]

    mmd = (K_xx.sum() / (m * m)) + (K_yy.sum() / (n * n)) - (2 * K_xy.sum() / (m * n))
    return mmd

def compute_wasserstein_dispersion(points, num_ref=100, range_min=-4, range_max=4):
    """
    Wasserstein distance

    para:
    - points (Tensor): 1D PyTorch shape: (N,)
    - num_ref (int): random sampling
    - range_min, range_max: range of desing

    return:
    - Wasserstein distance (Tensor)
    """
    points = points.view(-1).detach().cpu().numpy()  # convet NumPy 

    ref_samples = np.linspace(range_min, range_max, num_ref)  # uniform distrbution
    # ref_samples = np.array([2.309,-2.309,2,-2,2.1,-2.1,2.2,-2.2,2.4,-2.4,2.5,-2.5])
    wass_dist = wasserstein_distance(points, ref_samples)  # compute Wasserstein distance

    return torch.tensor(wass_dist, dtype=torch.float32)  # return PyTorch tensor 

def compute_mmd_dispersion(points, num_ref=100, range_min=-4, range_max=4, sigma=1.0):
    """
    MMD distance between sample points and a uniform reference distribution.

    Parameters:
    - points (Tensor): 1D PyTorch tensor, shape: (N,)
    - num_ref (int): number of reference points
    - range_min, range_max: the range for uniform distribution
    - sigma (float): bandwidth for the RBF kernel

    Returns:
    - MMD distance (Tensor)
    """
    points = points.view(-1)  # Ensure it's 1D
    uniform_samples = torch.linspace(range_min, range_max, num_ref)

    mmd = compute_mmd(points, uniform_samples, sigma=sigma)
    return mmd

def read_csv_xi(file_path, values_name):
    res = pd.read_csv(file_path)

    grouped_data = res.groupby(['ex_idx'])
    
    values_matrix = []

    for order_id, group in grouped_data:

        value = torch.stack([torch.tensor(value_str) for value_str in group[values_name]])

      
        values_matrix.append(value)

    values_matrix = torch.stack(values_matrix)

    dispersion_matrix = torch.zeros_like(values_matrix) 
    for i in range(values_matrix.shape[0]):  # i = 0..19
        row = values_matrix[i]  # shape: [11]

        for j in range(values_matrix.shape[1]):  # j = 0..10 this is for in round_idx start from 0 in csv file
            if j == 0:
                dispersion_matrix[i, j] = 0.0  # 
            else:
                subset = row[1:j+1]  # 
                dispersion = compute_mmd_dispersion(subset)
                dispersion_matrix[i, j] = dispersion
                
                
        # for j in range(values_matrix.shape[1]):  # j = 0..10 this is for in round_idx start from 1 in csv file
        #     subset = row[:j+1]  # 
        #     dispersion = compute_mmd_dispersion(subset)
        #     dispersion_matrix[i, j] = dispersion
    
    # zeros_col = torch.zeros((dispersion_matrix.shape[0], 1))  # shape: [20, 1]
    # dispersion_matrix = torch.cat([zeros_col, dispersion_matrix], dim=1)  # concat along columns

    return  dispersion_matrix

def read_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    algorithms = ['random', 'boed', 'dad']
    data_types = ['well', 'mis']

    def extract_lambda_val(filename):
        match = re.search(r'_lambda_([0-9.]+)', filename)
        if match:
            return float(match.group(1))
        return None  

    sorted_files = []

    for alg in algorithms:
        for dtype in data_types:
            matching_files = [f for f in csv_files if alg in f and dtype in f]

            no_lambda = [f for f in matching_files if extract_lambda_val(f) is None]
            with_lambda = [f for f in matching_files if extract_lambda_val(f) is not None]

    
            with_lambda.sort(key=lambda x: -extract_lambda_val(x))

            
            ordered = no_lambda + with_lambda
            sorted_files.extend(ordered)


    base_paths = [os.path.join(directory, f) for f in sorted_files]
    labels = [f[:-4] for f in sorted_files]


    return base_paths, labels


    



