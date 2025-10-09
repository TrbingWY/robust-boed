import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def load_true_theta(file_path):
    true_thetas = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines) - 1:
            true_theta_line = 'torch.tensor(' + lines[i].strip() + lines[i+1].strip() + ')'
            true_theta = eval(true_theta_line, {'torch': torch})
            true_thetas.append(true_theta)
            i += 2
    return true_thetas

def save_csv(output, iteration_flag, save_path, save_path_txt):
    res = pd.concat(output)
    updated_file_path = f"{save_path}_{iteration_flag}.csv"
    # updated_file_path = save_path + str(iteration_flag)
    save_path_txt = save_path_txt + str(iteration_flag)
    print("save to csv", updated_file_path)
    res.to_csv(updated_file_path, index=False)
