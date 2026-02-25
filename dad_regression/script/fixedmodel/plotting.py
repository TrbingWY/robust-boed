import ast
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from matplotlib import font_manager
import pandas as pd
import re

color_box = ['blue', 'red', 'green', 'purple', 'orange', 'cyan','yellow','c']*3
shape_box = ['o', 's', 'D', '^', 'v', 'x','p','>','<','+','*']*3
FontSize = 20



def plot_generalization_error_vs_round(dir, generalization_errors,labels):
    """
    Plot generalization error vs. number of rounds.

    Args:
        generalization_errors (list): List of generalization errors for each round.
    """
    # num_rounds = len(generalization_errors[0])
    # rounds = list(range(num_rounds+1))
    num_rounds = generalization_errors[0].shape[1] #shape:(ex_num_rounds, T)
    rounds = list(range(num_rounds))
    
    plt.figure(figsize=(8, 6))
    for i, (generalization_error, label,color, shape) in enumerate(zip(generalization_errors,labels,color_box,shape_box)):
        mean_values = torch.mean(generalization_error, dim=0)
        standard_errors = torch.std(generalization_error, dim=0) / np.sqrt(generalization_error.shape[0])
        plt.plot(rounds, mean_values, marker=shape,
                linestyle='-',
                color=color, label=label)
        
        plt.fill_between(rounds, mean_values - standard_errors, mean_values + standard_errors, alpha=0.3)


    # plt.yscale("log") 
    plt.tick_params(axis='both', which='major', labelsize=FontSize)  
    plt.xlabel("Number of Rounds",fontsize = FontSize)
    plt.ylabel("Generalization Error",fontsize = FontSize)
    plt.title("Generalization Error vs. Number of Rounds",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(dir + "generalization errors.png")
    print(f"plot save to {dir} generalization errors.png")


def plot_covariet_shift_vs_round(dir, covariate_shifts,labels,method="wass"):
    """
    Plot generalization error vs. number of rounds.

    Args:
        generalization_errors (list): List of generalization errors for each round.
    """
    # num_rounds = len(covariate_shifts[0])
    # rounds = list(range(num_rounds+1))
    num_rounds = covariate_shifts[0].shape[1] #shape:(ex_num_rounds, T)
    rounds = list(range(1,num_rounds))

    
    plt.figure(figsize=(8, 6))
    for i, (covariate_shift, label,color, shape) in enumerate(zip(covariate_shifts,labels,color_box,shape_box)):
        mean_values = torch.mean(covariate_shift, dim=0).detach().cpu().numpy()
        standard_errors = (torch.std(covariate_shift, dim=0) / np.sqrt(covariate_shift.shape[0])).detach().cpu().numpy()
        plt.plot(rounds, mean_values[1:], marker=shape,
                linestyle='-',
                color=color, label=label)
        
        plt.fill_between(rounds, mean_values[1:] - standard_errors[1:], mean_values[1:] + standard_errors[1:], alpha=0.3)

    plt.tick_params(axis='both', which='major', labelsize=FontSize)  
    plt.xlabel("Number of Rounds",fontsize = FontSize)
    plt.ylabel("Maximum Mean Discrepance",fontsize = FontSize)
    plt.title("Maximum Mean Discrepance vs. Number of Rounds",fontsize = FontSize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig( dir +"covariate_shifts_"+method+" .png")
    print(f"plot save to {dir} covariate_"+method+" .png")

def parse_tensor_string(tensor_str):
    
    tensor_str = tensor_str.strip()
    
    tensor_str = re.sub(r'\s+', ' ', tensor_str)
   
    tensor_str = re.sub(r'\s*\n\s*', ', ', tensor_str)  
    tensor_str = re.sub(r'(?<=\d)\s+(?=\d)', ', ', tensor_str)  
    tensor_str = re.sub(r'\[\s+', '[', tensor_str)  
    tensor_str = re.sub(r'\s+\]', ']', tensor_str)  
    tensor_str = re.sub(r' ', ',', tensor_str)
   
    tensor_str = re.sub(r',\s*,', ', ', tensor_str)

    if not tensor_str.startswith("tensor") and not tensor_str.startswith("torch.tensor"):
        tensor_str = f"torch.tensor({tensor_str})"

    try:
        result = eval(tensor_str, {'torch': torch, 'tensor': lambda x: x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)})
    except Exception as e:
        print(f"Error parsing tensor string: {tensor_str}")
        raise e

    return result

def read_csv(file_path):
    res = pd.read_csv(file_path)

    grouped_data = res.groupby(['ex_idx'])
    
    covarite_shift_degrees = []
    generalization_errors = []
    Gerror_matrix = []

    for order_id, group in grouped_data:

        covariate_shift_degree_value = torch.stack([parse_tensor_string(rmse_str) for rmse_str in group['covariate_shift_degree']])
        generalization_error_values = torch.stack([parse_tensor_string(nll_str) for nll_str in group['generalization_error']])
      
        generalization_errors.append(generalization_error_values)
        covarite_shift_degrees.append(covariate_shift_degree_value)

    generalization_errors = torch.stack(generalization_errors)
    covarite_shift_degrees = torch.stack(covarite_shift_degrees)
    
    return generalization_errors, covarite_shift_degrees

def read_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    algorithms = ['random', 'boed', 'dad']
    data_types = ['well', 'mis']

    def extract_lambda_val(filename):
        match = re.search(r'_lambda_([0-9.]+)', filename)
        if match:
            return float(match.group(1))
        return None  # 没有 lambda 的返回 None

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

def compare_different_mis(
    mis_dirs: dict,
    output_path: str,
    mis_levels=(0, 1, 2),
    random_prefix: str = "random",
    ridea_prefix: str = "db_boed",   # R-IDeA
    bad_prefix: str = "boed",        # BAD / R-I
    ratio_mode: str = "random_over_method",  # or "method_over_random"
    normalize_to_no_mis: bool = True,
    show_center_line: bool = True,   # center line uses median
):
    """
    Plot ratio bounds across misspecification levels with min-max band (no averaging).

    ratio_mode:
        - "random_over_method":  ratio = Random / Method  (larger is better)
        - "method_over_random":  ratio = Method / Random  (smaller is better)
    """

    def find_csv(directory, prefix):
        files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
        matches = [f for f in files if f.lower().startswith(prefix.lower())]
        if len(matches) == 0:
            raise FileNotFoundError(f"No file starting with '{prefix}' in {directory}")
        matches = sorted(matches)
        return os.path.join(directory, matches[0])

    def final_round_values(csv_path):
        G, _ = read_csv(csv_path)   # reuse your existing function
        G = G.squeeze()
        if G.ndim == 1:
            vals = G
        else:
            vals = G[..., -1]       # last round per ex
        return vals.detach().cpu().numpy()

    # enforce x-axis only has [0,1,2]
    mis_levels = list(mis_levels)
    for m in mis_levels:
        if m not in mis_dirs:
            raise KeyError(f"mis_dirs missing key {m}. Provided keys: {sorted(mis_dirs.keys())}")

    if ratio_mode not in ("random_over_method", "method_over_random"):
        raise ValueError("ratio_mode must be 'random_over_method' or 'method_over_random'")

    x = np.array(mis_levels, dtype=float)

    ratio_ridea_list = []
    ratio_bad_list = []

    for mis in mis_levels:
        d = mis_dirs[mis]

        random_csv = find_csv(d, random_prefix)
        ridea_csv  = find_csv(d, ridea_prefix)
        bad_csv    = find_csv(d, bad_prefix)

        r = final_round_values(random_csv)
        i = final_round_values(ridea_csv)
        b = final_round_values(bad_csv)

        n = min(len(r), len(i), len(b))
        r, i, b = r[:n], i[:n], b[:n]

        if ratio_mode == "random_over_method":
            ratio_ridea = r / (i + 1e-12)
            ratio_bad   = r / (b + 1e-12)
            label_ridea = "Random / R-IDeA"
            label_bad   = "Random / BAD"
            ylabel = "Generalization error ratio"
        else:
            ratio_ridea = i / (r + 1e-12)
            ratio_bad   = b / (r + 1e-12)
            label_ridea = "R-IDeA / Random"
            label_bad   = "BAD / Random"
            ylabel = "Generalization error ratio"

        ratio_ridea_list.append(ratio_ridea)
        ratio_bad_list.append(ratio_bad)

        print(f"[mis={mis}] n_ex={n}  "
              f"R-IDeA ratio min={ratio_ridea.min():.3f} max={ratio_ridea.max():.3f}  "
              f"BAD ratio min={ratio_bad.min():.3f} max={ratio_bad.max():.3f}")

    # normalize so that mis=0 center is 1 (using median, not mean)
    if normalize_to_no_mis:
        idx0 = mis_levels.index(0)
        base_ridea = np.median(ratio_ridea_list[idx0])
        base_bad   = np.median(ratio_bad_list[idx0])

        ratio_ridea_list = [arr / (base_ridea + 1e-12) for arr in ratio_ridea_list]
        ratio_bad_list   = [arr / (base_bad + 1e-12) for arr in ratio_bad_list]

    # 计算 bounds
    def bounds_and_center(arr_list):
        lowers = np.array([np.min(a) for a in arr_list], dtype=float)
        uppers = np.array([np.max(a) for a in arr_list], dtype=float)
        centers = np.array([np.median(a) for a in arr_list], dtype=float)

        lower_err = centers - lowers
        upper_err = uppers - centers
        yerr = np.vstack([lower_err, upper_err])

        return centers, yerr

    med_i, yerr_i = bounds_and_center(ratio_ridea_list)
    med_b, yerr_b = bounds_and_center(ratio_bad_list)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))

    plt.errorbar(
        x, med_i, yerr=yerr_i,
        fmt='o', capsize=6, 
        elinewidth=2,
        markersize=8,
        label=label_ridea
    )

    plt.errorbar(
        x, med_b, yerr=yerr_b,
        fmt='s', capsize=6,
        elinewidth=2,
        markersize=8,
        label=label_bad
    )

    plt.axhline(1.0, linestyle='--', linewidth=1)

    plt.xticks(x, [str(int(v)) for v in x], fontsize=16)
    plt.xlabel("Degree of misspecification", fontsize=18)
    plt.ylabel("Generalization error ratio", fontsize=18)
    plt.title("Generalization error ratio vs. misspecification", fontsize=18)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":

    directory = "./toy_examples/dad_regression/results/test05/result42_final/"

    directory = "/mnt/iusers01/sk01/j43716rt/scratch/projects/robust-boed/dad_regression/results/mis3/tau/lambda1/"
    output_dir = directory

    csv_files = sorted(
        # [f for f in os.listdir(directory) if f.endswith('.csv')],
        [
        f for f in os.listdir(directory)
        if f.endswith('.csv') and 'well' not in f
        ],
        # key=lambda f: (
        #     ['random', 'boed', 'dad'].index(next((x for x in ['random', 'boed', 'dad'] if x in f), 'zzz')),
        #     1 if '_mis' in f else 0
        # )
        #   key=lambda f: (
        #     ['random', 'boed_mis', 'dad','1_mis','0.5_mis','0.25_mis'].index(next((x for x in ['random', 'boed_mis', 'dad','1_mis','0.5_mis','0.25_mis'] if x in f), 'zzz')),
        #     1 if '_mis' in f else 0
        # )
    )
    base_paths = [os.path.join(directory, f) for f in csv_files]

    # base_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    # print(base_paths)

    # csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    labels = [re.sub(r'_+', '_', re.sub(r'\d+', '', f[:-4])).strip('_') for f in csv_files]


    labels = [f[:-4].strip('_') for f in csv_files]

    # Replace 'boed' with 'bad'
    labels = [label.replace("boed", "BAD") for label in labels]

    labels = [label.replace("random", "Random") for label in labels]

    labels = [label.replace("dad", "DAD") for label in labels]

    # Special handling for lambda cases
    for i, f in enumerate(csv_files):
        if "boed_lambda_1" in f:
            labels[i] = "BAD-Adj and λ=1"
        elif "boed_lambda_0.5" in f:
            labels[i] = "BAD-Adj and λ=0.5"
        elif "boed_lambda_0.25" in f:
            labels[i] = "BAD-Adj and λ=0.25"

    print(labels)


    # base_paths, labels = read_files(directory)

    generalization_errors, covarite_shift_degrees = zip(*[read_csv(base_path) for base_path in base_paths])

    # print(font_prop.get_name())
    plot_generalization_error_vs_round(output_dir,generalization_errors,labels)
    plot_covariet_shift_vs_round(output_dir,covarite_shift_degrees, labels,method = "mmd") # this covariate shift is in the files instead of computing mmd/wass by xi

    mis_dirs = {
    0: "./compare/well/",
    1: "./compare/mis2/",
    2: "./dad_regression/results/test05/result42_final_v5/newMethod/compare/mis3/",
    }

    compare_different_mis(
        mis_dirs=mis_dirs,
        output_path="./compare/ratio_vs_mis.png",
        ratio_mode="method_over_random",  #random_over_method;method_over_random
        mis_levels=(0, 1, 2),
    )


  