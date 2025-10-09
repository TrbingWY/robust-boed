# main.py
import os
import argparse
import pandas as pd
from utils import load_true_theta, save_csv
from runner import single_run
import pyro
import torch
import gc
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="PK location inference via VBOED")
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--num_experiments", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_steps", default=100, type=int)
    parser.add_argument("--grids_num", default=200, type=int)
    parser.add_argument("--pk_method", default="db_boed_lambda_0.5_tau_2", type=str)
    parser.add_argument("--biased_sample_flag", default=False, type=bool)
    parser.add_argument("--num_parallel", default=1, type=int) #different seed
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--dir", default="../results/db_boed/tmp/", type=str)
    parser.add_argument("--misspecification_flag", type=str, default="False")
    # design space(hour)
    parser.add_argument("--t_min", default=0.0, type=float)
    parser.add_argument("--t_max", default=24.0, type=float)
    parser.add_argument("--t_step", default=0.1, type=float)
    parser.add_argument("--eig_method", default="nmc_eig", type=str) #db_boed_lambda_0.5_tau_20"
    return parser.parse_args()


def setup_save_dir(args):
    mode = "biased" if args.biased_sample_flag else "unbiased"
    save_dir = f"{args.dir}/{mode}_design_{args.num_parallel}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/tmp/", exist_ok=True)
    return save_dir


def main():
    args = parse_args()
    print(vars(args))
    args.misspecification_flag = args.misspecification_flag.lower() in ("true", "1", "yes", "True")

    save_dir = setup_save_dir(args)

    suf = f"{args.pk_method}_{args.grids_num}" + ("_miss" if args.misspecification_flag else "")
    save_path = os.path.join(save_dir, suf)
    save_path_txt = os.path.join(save_dir, suf)
    save_path_tmp = os.path.join(save_dir, "tmp", suf)
    save_path_txt_tmp = os.path.join(save_dir, "tmp", suf)

   
    theta_list = [torch.tensor([[1.5, 0.15, 15.0]]).log()]
   
    output = []

    for theta_idx, true_theta_log in enumerate(theta_list):
        for run_id in range(args.num_parallel):
            pyro.clear_param_store()
            if args.seed >= 0:
                seed = args.seed + run_id * 10 + theta_idx * 100
                pyro.set_rng_seed(seed)
                torch.manual_seed(seed)
            else:
                seed = int(torch.rand(tuple()) * 2 ** 30)
                pyro.set_rng_seed(seed)
                torch.manual_seed(seed)

            print(f"=== Run ID {run_id} ===")
            true_theta_log = true_theta_log.squeeze(0)
            print("true_theta_log:", true_theta_log)
        

            xis_all, run_mean, run_std, rmse_all, run_nll, ge_all,mmd_all = single_run(
                num_candidates=args.grids_num,
                eig_method=args.eig_method,
                dir=save_dir,
                num_experiments=args.num_experiments,
                alg_method=args.pk_method,
                true_theta=true_theta_log,                #log-space
                biased_sample_flag=args.biased_sample_flag,
                run_id_tensors=torch.tensor([[1.5, 0.15, 15.0]]).log(),
                misspecification_flag=args.misspecification_flag,
                design_space=(args.t_min, args.t_max, args.t_step)
            )

            output.append(pd.DataFrame({
                'run_id': run_id + 1,
                'theta_idx': theta_idx,
                'xis': xis_all,
                'mean': run_mean,
                'std': run_std,
                'rmse': rmse_all,
                'order': list(range(len(run_std))),
                'nll': run_nll,
                'Gerror': ge_all,
                'DesignMMD': mmd_all,
            }))

            if (run_id + 1) % 2 == 0:
                save_csv(output, run_id, save_path_tmp, save_path_txt_tmp)

    save_csv(output, run_id, save_path, save_path_txt)


if __name__ == "__main__":
    main()