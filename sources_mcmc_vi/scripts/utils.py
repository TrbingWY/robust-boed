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
    save_path_txt = f"{save_path_txt}_{iteration_flag}.txt"
    # updated_file_path = save_path + str(iteration_flag)
    save_path_txt = save_path_txt + str(iteration_flag)
    print("save to csv", updated_file_path)
    res.to_csv(updated_file_path, index=False)

    grouped_data = res.groupby(['order'])
    for order_id, group in grouped_data:
        rmse_tensors = torch.stack(list(group['rmse']))
        nll_tensors = torch.stack(list(group['nll']))
        tmp_rmse = torch.mean(rmse_tensors, dim=0)
        tmp_nll = torch.mean(nll_tensors)

        with open(save_path_txt, "a+") as f:
            f.write(f"{order_id}: {tmp_rmse}\n")
        with open(save_path_txt + "nll", "a+") as f:
            f.write(f"{order_id}: {tmp_nll}\n")

    print("Saved mean RMSE and NLL for iteration", iteration_flag)

def plot_generalization_error_vs_round(dir, generalization_errors):
    num_rounds = len(generalization_errors)
    rounds = list(range(1, num_rounds + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, generalization_errors, marker="o", linestyle="-", label="Generalization Error")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Generalization Error (MSE)")
    plt.title("Generalization Error vs. Number of Rounds")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "generalization_errors.png"))
