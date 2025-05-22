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

def read_csv_xi(file_path, values_name):
    res = pd.read_csv(file_path)

    res['ex_idx'] = res['theta_idx'].astype(str) + "_" + res['run_id'].astype(str)
    grouped_data = res.groupby(['ex_idx'])

    values_matrix = []

    for _, group in grouped_data:
        values = [parse_tensor_string(s) for s in group[values_name]]
        values_tensor = torch.stack(values)  # shape: [num_rounds, 2]
        values_matrix.append(values_tensor)

    values_matrix = torch.stack(values_matrix)  # shape: [num_experiments, num_rounds, 2]

    # compute dispersion
    dispersion_matrix = torch.zeros(values_matrix.shape[:2])  # shape: [num_experiments, num_rounds]

    for i in range(values_matrix.shape[0]):  
        row = values_matrix[i]  # shape: [num_rounds, 2]

        for j in range(row.shape[0]):
            if j == 0:
                dispersion_matrix[i, j] = 0.0
            else:
                subset = row[:j+1]  # shape: [j+1, 2]
                dispersion = compute_mmd_dispersion(subset)
                dispersion_matrix[i, j] = dispersion

    return dispersion_matrix

def plot_covariate_shift_vs_round(output_dir, xis, labels):
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

    plt.xlabel("Number of Rounds",fontsize = FontSize)
    plt.ylabel("Maximum mean discrepancy ",fontsize = FontSize)
    plt.title("Maximum mean discrepancy vs. Number of Rounds",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize = FontSize)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "covariate_shift.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
      
def initialize_candidate_designs(num_candidates=200, seed=777):
    ranges = [torch.arange(-4, 4.1, 0.1) for _ in range(2)]
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

def compute_mmd_dispersion(points):
    """
    MMD distance between sample points and a uniform reference distribution.

    Parameters:
    - points (Tensor): 2D PyTorch tensor, shape: (N,2)
    - sigma (float): bandwidth for the RBF kernel

    Returns:
    - MMD distance (Tensor)
    """
    points = points.view(-1,2)  # Ensure it's 1D
    uniform_samples = initialize_candidate_designs()

    mmd = compute_mmd(points, uniform_samples, sigma=1.0)
    return mmd

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

def plot_generalization_error_vs_round(dir, generalization_errors,labels):
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

    # plt.yscale("log") 
    plt.xlabel("Number of Rounds",fontsize = FontSize)
    plt.ylabel("Generalization Error",fontsize = FontSize)
    plt.title("Generalization Error vs. Number of Rounds",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize = FontSize)
    plt.tight_layout()
    plt.savefig(dir + "generalization errors_wass.png")
    print(f"plot save to {dir} generalization errors.png")


def plot_xyrmse_vs_round(output_dir, rmses, labels):
    num_rounds = rmses[0].shape[1]  # shape: (num_experiments, num_rounds, 2)
    rounds = list(range(num_rounds))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6)) 

    directions = ["x", "y"]
    for dim in range(2):
        ax = axes[dim]
        for i, (rmse, label, color, shape) in enumerate(zip(rmses, labels, color_box, shape_box)):
            rmse_dim = rmse[:, :, dim]  # shape: [num_experiments, num_rounds]
            mean_values = torch.mean(rmse_dim, dim=0)
            standard_errors = torch.std(rmse_dim, dim=0) / np.sqrt(rmse_dim.shape[0])

            ax.plot(rounds, mean_values, marker=shape, linestyle='-', color=color, label=label)
            ax.fill_between(rounds, mean_values - standard_errors, mean_values + standard_errors, alpha=0.3)

        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel(f"RMSE ({directions[dim]} direction)")
        ax.set_title(f"RMSE vs. Rounds ({directions[dim]}-axis)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "rmse_xy_subplots.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__=="__main__":
    # subset = torch.tensor([[-1.0, -2.0], [0.0, 0.0], [2.0, 3.0]])  # shape: [3, 2]
    # disp = compute_mmd_dispersion(subset)
    # print("Dispersion:", disp.item())

    directory =  "./toy_examples/sources_mcmc_vi/results/source_final/empirical/"
    directory = "./toy_examples/sources_mcmc_vi/results/source_final/newmethod/"
    

    output_dir = directory

    csv_files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.csv')],
        key=lambda f: (
            ['random', 'boed', 'dad'].index(next((x for x in ['random', 'boed', 'dad'] if x in f), 'zzz')),
            1 if '_mis' in f else 0
        )
    )

    base_paths = [os.path.join(directory, f) for f in csv_files]

    # labels = [re.sub(r'_+', '_', re.sub(r'\d+', '', f[:-4])).strip('_') for f in csv_files]

    labels = [f[:-4].strip('_') for f in csv_files]

    # Replace 'boed' with 'bad'
    labels = [label.replace("boed", "BAD") for label in labels]

    labels = [label.replace("random", "Random") for label in labels]

    labels = [label.replace("dad", "DAD") for label in labels]

    # Special handling for lambda cases
    for i, f in enumerate(csv_files):
        if "boed_new_lambda1" in f:
            labels[i] = "BAD-Adj and λ=1"
        elif "boed_new_lambda05" in f:
            labels[i] = "BAD-Adj and λ=0.5"
        elif "boed_new_lambda025" in f:
            labels[i] = "BAD-Adj and λ=0.25"

    print(base_paths)

    print(labels)

    generalization_errors,xis, rmses = zip(*[read_csv(base_path) for base_path in base_paths])

    plot_generalization_error_vs_round(output_dir,generalization_errors,labels)

    plot_covariate_shift_vs_round(output_dir, xis, labels)
