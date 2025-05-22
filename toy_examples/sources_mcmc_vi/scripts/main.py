import os
import argparse
import pandas as pd
from utils import load_true_theta, save_csv
from runner import single_run
import pyro
import torch
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Location inference via VBOED")
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--num_experiments", default=30, type=int)
    parser.add_argument("--num_sources", default=2, type=int)
    parser.add_argument("--noise_scale", default=0.1, type=float)
    parser.add_argument("--base_signal", default=0.1, type=float)
    parser.add_argument("--max_signal", default=1e-4, type=float)
    parser.add_argument("--num_dim", default=2, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_steps", default=100, type=int)
    parser.add_argument("--grids_num", default=200, type=int)
    parser.add_argument("--location_method", default="boed_new_025", type=str)
    parser.add_argument("--biased_sample_flag", default=False, type=bool)
    parser.add_argument("--num_parallel", default=1, type=int)
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--dir",default= "../results/", type= str)
    parser.add_argument("--misspecification_flag", type=str, default="True")
    parser.add_argument("--experiment_id", type=str, default="True")
    parser.add_argument("--run_id", type=str, default="True")
    
    return parser.parse_args()


def setup_save_dir(args):
    mode = "biased" if args.biased_sample_flag else "unbiased"
    save_dir = f"{args.dir}/{mode}_design_{args.num_parallel}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+"/tmp/", exist_ok=True)
    return save_dir


def main():
    args = parse_args()
    print(vars(args))
    args.misspecification_flag = args.misspecification_flag.lower() in ("true", "1", "yes","True")

    save_dir = setup_save_dir(args)

    if not args.misspecification_flag:
        save_path = os.path.join(save_dir, f"{args.location_method}_{args.grids_num}")
        save_path_txt = os.path.join(save_dir, f"{args.location_method}_{args.grids_num}")

        save_path_tmp = os.path.join(save_dir+"/tmp/", f"{args.location_method}_{args.grids_num}")
        save_path_txt_tmp = os.path.join(save_dir+"/tmp/", f"{args.location_method}_{args.grids_num}")
    else:
        save_path = os.path.join(save_dir, f"{args.location_method}_{args.grids_num}_miss")
        save_path_txt = os.path.join(save_dir, f"{args.location_method}_{args.grids_num}_miss")

        save_path_tmp = os.path.join(save_dir+"/tmp/", f"{args.location_method}_{args.grids_num}_miss")
        save_path_txt_tmp = os.path.join(save_dir+"/tmp/", f"{args.location_method}_{args.grids_num}_miss")

    theta_list = load_true_theta("./true_thetas_500.txt")

    model_location = f"./mlruns/{args.experiment_id}/{args.run_id}/artifacts/model"

    output = []
    # torch.use_deterministic_algorithms(True)

    for theta_idx, true_theta in enumerate(theta_list[:100]):
        for run_id in range(args.num_parallel): 
            
            pyro.clear_param_store()
            if args.seed >= 0:
                pyro.set_rng_seed(args.seed+ run_id*10 + theta_idx *100)
                torch.manual_seed(args.seed+ run_id*10 + theta_idx*100) 
            else:
                seed = int(torch.rand(tuple()) * 2 ** 30)
                pyro.set_rng_seed(seed)
                torch.manual_seed(seed)

            print(f"=== Run ID {run_id} ===")
            print("true_theta:",true_theta)

            xis_all, run_mean, run_std, rmse_all, run_nll, generalization_error_all = single_run(
                p=args.num_dim,
                K=args.num_sources,
                noise_scale=args.noise_scale,
                num_candidates=args.grids_num,
                eig_method="nmc_eig",
                dir="../results/boed/",
                num_experiments=args.num_experiments,
                alg_method=args.location_method,
                true_theta=true_theta,
                biased_sample_flag=args.biased_sample_flag,
                misspecification_flag = args.misspecification_flag,
                model_location= model_location,
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
                'Gerror': generalization_error_all,
            }))

            if (theta_idx + 1) % 10 == 0:
                save_csv(output, theta_idx, save_path_tmp, save_path_txt_tmp)
    save_csv(output, theta_idx, save_path, save_path_txt)


if __name__ == "__main__":
    main()
