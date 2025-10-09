import os
import re
import pandas as pd
import torch
import ast

import matplotlib.pyplot as plt
import numpy as np


color_box = ['blue', 'red', 'green', 'purple', 'orange', 'cyan','yellow','c']*3
shape_box = ['o', 's', 'D', '^', 'v', 'x','p','>','<','+','*']*3
FontSize = 20



def parse_tensor_string(tensor_str):
    """Parse a string like 'tensor([1.0, 2.0])' or '[1.0, 2.0]' to torch.Tensor"""
    try:
        # Remove 'tensor(' and ')' if present
        if 'tensor' in tensor_str:
            tensor_str = tensor_str.replace('tensor(', '').rstrip(')')
        return torch.tensor(ast.literal_eval(tensor_str), dtype=torch.float32)
    except Exception as e:
        print(f"Error parsing tensor: {tensor_str}")
        raise e

def plot_covariate_shift_vs_round(output_dir, xis, labels,TMP):
    plt.figure(figsize=(8,6))
    
    for index, (xi, label, color, shape) in enumerate(zip(xis, labels, color_box, shape_box)):
        dispersion_matrix = torch.zeros(xi.shape[0], xi.shape[1]) 

        for i in range(xi.shape[0]):  # num_experiments
            row = xi[i]
            for j in range(xi.shape[1]):
                if j == 0:
                    dispersion_matrix[i, j] = 0.0
                else:
                    subset = row[1:j+1]
                    dispersion = compute_mmd_dispersion(subset)
                    dispersion_matrix[i, j] = dispersion

        mean_values = torch.mean(dispersion_matrix, dim=0)
        standard_errors = torch.std(dispersion_matrix, dim=0) / np.sqrt(dispersion_matrix.shape[0])
        
        rounds = list(range(xi.shape[1]))
        plt.plot(rounds[1:], mean_values[1:], marker=shape, linestyle='-', color=color, label=label,markersize=10)
        plt.fill_between(rounds[1:], mean_values[1:] - standard_errors[1:], mean_values[1:] + standard_errors[1:], alpha=0.3, color=color)

    plt.xlabel("Number of Training Samples",fontsize = FontSize)
    plt.ylabel("Maximum mean discrepancy ",fontsize = FontSize)
    plt.title("MMD under " + TMP+" \n — Pharmacokinetic model",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize = FontSize-6)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "covariate_shift.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
       
def initialize_candidate_designs(num_candidates=200, seed=777):
    ranges = [torch.arange(0.0, 24.0, 0.1) for _ in range(2)]
    grid = torch.cartesian_prod(*ranges)
    torch.manual_seed(seed)
    indices = torch.randperm(len(grid))[:num_candidates]
    candidate_designs = grid[indices]
    return candidate_designs

def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between two sets of points.
    """
    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)

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

def compute_mmd_dispersion(points, design_space=(0.0, 24.0), sigma=1.0, num_ref=2000):
    pts = torch.as_tensor(points, dtype=torch.float32)

    # Normalize to (N, D)
    if pts.ndim == 0:
        pts = pts.reshape(1, 1)
    elif pts.ndim == 1:
        pts = pts[:, None]
    elif pts.ndim == 3 and pts.shape[-1] == 1:
        pts = pts.squeeze(-1)
        if pts.ndim == 1:
            pts = pts[:, None]
    elif pts.ndim != 2:
        pts = pts.reshape(pts.shape[0], -1)

    N, D = pts.shape

    # Build uniform reference with matching dimensionality
    low, high = design_space
    if D == 1:
        ref = low + (high - low) * torch.rand(num_ref, 1)
    elif D == 2:
        # Use your existing helper if available; otherwise sample IID
        try:
            ref = initialize_candidate_designs(num_candidates=num_ref)
        except NameError:
            ref = low + (high - low) * torch.rand(num_ref, 2)
    else:
        ref = low + (high - low) * torch.rand(num_ref, D)

    return compute_mmd(pts, ref, sigma=sigma)

def read_csv(file_path):
    res = pd.read_csv(file_path)

    grouped_data = res.groupby(['run_id'])
    
    generalization_errors = []
    xis = []
    rmses= []
    for _, group in grouped_data:
        group_theta = group.groupby(['theta_idx'])
        for order_id, group_t in group_theta:
     
            generalization_error_values = torch.stack([torch.tensor(nll_str) for nll_str in group_t['Gerror']])
            xi = torch.stack([parse_tensor_string(xi_str) for xi_str in group_t["xis"]])
            rmse = torch.stack([parse_tensor_string(rmse_str) for rmse_str in group_t["rmse"]])

            generalization_errors.append(generalization_error_values)
            xis.append(xi)
            rmses.append(rmse)
      

    generalization_errors = torch.stack(generalization_errors)
    xis = torch.stack(xis)
    rmses = torch.stack(rmses)

    
    return generalization_errors, xis, rmses #shape [run_id_num * theta_idx_num,31][100,31,2] [100,31,2]

def plot_generalization_error_vs_round(dir, generalization_errors,labels,TMP):
    """
    Plot generalization error vs. number of rounds.

    Args:
        generalization_errors (list): List of generalization errors for each round.
    """
    # num_rounds = len(generalization_errors[1])
    # rounds = list(range(num_rounds+1))
    num_rounds = generalization_errors[0].shape[1] #shape:(ex_num_rounds, T)
    rounds = list(range(num_rounds))
    
    plt.figure(figsize=(8, 6))
    for i, (generalization_error, label,color, shape) in enumerate(zip(generalization_errors,labels,color_box,shape_box)):
        mean_values = torch.mean(generalization_error, dim=0)
        standard_errors = torch.std(generalization_error, dim=0) / np.sqrt(generalization_error.shape[0])
        
        plt.plot(rounds, mean_values, marker=shape,
                linestyle='-',
                color=color, label=label,markersize=10)
        
        plt.fill_between(rounds, mean_values - standard_errors, mean_values + standard_errors, alpha=0.3)
        print(mean_values,standard_errors)

    # plt.yscale("log") 
    plt.xlabel("Number of Training Samples",fontsize = FontSize)
    plt.ylabel("Generalization Error",fontsize = FontSize)
    plt.title("Generalization Error under "+ TMP + " \n — Pharmacokinetic model",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize = FontSize-6)
    plt.tight_layout()
    plt.savefig(dir + "generalization_errors.png")
    print(f"plot save to {dir} generalization_errors.png")



if __name__=="__main__":
    subset = torch.tensor([[-1.0, -2.0], [0.0, 0.0], [2.0, 3.0]])  # shape: [3, 2]
    disp = compute_mmd_dispersion(subset)
    print("Dispersion:", disp.item())

    directory = ""
   
    output_dir = directory
    TMP = "model misspecfication"


    base_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    # print(base_paths)

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    labels = [re.sub(r'_+', '_', re.sub(r'\d+', '', f[:-4])).strip('_') for f in csv_files]

    

    generalization_errors,xis, rmses = zip(*[read_csv(base_path) for base_path in base_paths])

    plot_generalization_error_vs_round(output_dir,generalization_errors,labels,TMP= TMP)

    plot_covariate_shift_vs_round(output_dir, xis, labels,TMP= TMP)
