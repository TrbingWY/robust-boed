import torch
from polynomial_model import PolynomialModel
from boed_experiment import BOEDExperiment
from utils import compute_covarite_shift
from plotting import plot_generalization_error_vs_round,plot_covariet_shift_vs_round
import os,sys
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import MCMC, NUTS
import pandas as pd
import gc
import mlflow
import mlflow.pytorch
from oed.design import OED
from oed.primitives import observation_sample, latent_sample, compute_design

def test_phase(model, posterior_mean, posterior_sd, test_x,xis,ys, ax, round,misspecified_flag,PLOT):
    """
    Evaluate the generalization error of the model using test data.
    barf(x) = -1.6720 + 2.0000 * x
    Args:
        model (PolynomialModel): The model used for generating synthetic data.
        posterior_mean (torch.Tensor): The posterior mean inferred from training.
        posterior_sd (torch.Tensor): The posterior standard deviation.
        
        test_x (int): test points
        

    Returns:
        generalization_error (list): List of errors for each test point.
    """

    mean_gen_error, mean_bias2, mean_variance, mean_cross = compute_error_decomposition(
    model, posterior_mean, test_x, barf_params=(-1.6720, 2.0000)
)

    print(f"Decomposed Gen Error: {mean_gen_error:.2f} = Bias² {mean_bias2:.2f} + Variance {mean_variance:.2f} + Cross {2*mean_cross:.2f}")

    return mean_gen_error, mean_bias2, mean_variance, mean_cross

def compute_error_decomposition(model, posterior_mean, test_x, barf_params):
    """
    Decompose generalization error into bias², variance, cross term.
    """
    # f_star(x)
    f_star_x = model.synthetic_data(test_x)

    # f(x): current model
    f_x = sum(posterior_mean[i].item() * test_x**i for i in range(len(posterior_mean)))

    # barf(x): fixed average model (e.g., least squares)
    a_barf, b_barf = barf_params
    barf_x = a_barf + b_barf * test_x  # assuming linear barf(x)

    # Compute terms
    bias2 = (barf_x - f_star_x) ** 2
    variance = (f_x - barf_x) ** 2
    cross = (f_x - barf_x) * (barf_x - f_star_x)

    # Sum to get full generalization error
    gen_error = bias2 + variance + 2 * cross
    mean_gen_error = torch.mean(gen_error).item()
    mean_bias2 = torch.mean(bias2).item()
    mean_variance = torch.mean(variance).item()
    mean_cross = torch.mean(cross).item()

    return mean_gen_error, mean_bias2, mean_variance, mean_cross


def experiment(
        experiment_id,
        run_id,method = "boed",
        misspecified_flag = False, 
        eig_method = "marginal_eig",
        num_rounds = 5,
        num_ex =20,
        dir="./test/",
        covarite_shift_method= "wass",
        seed = -1,
        ):


    num_test_points = 100
    design_space =  (-4, 4)
    #design_space (tuple): The range of the design space (min, max).
    test_x = torch.linspace(*design_space, steps=num_test_points).unsqueeze(-1)
    
    fig, axes = plt.subplots(1, num_rounds+1, figsize=(num_rounds*6, 15))  # Adjust 6x5 grid for 30 rounds
    axes = axes.flatten()
    # Training

    generalization_error = torch.zeros(num_ex,num_rounds+1)
    covarite_shift_degrees = torch.zeros(num_ex,num_rounds+1)
    rmse_values = torch.zeros(num_rounds+1)

    output = []
    if num_ex == 1:
        PLOT = True
    else:
        PLOT = False
    


    for ex_idx in range(num_ex):
            # Initialize variables
        if not misspecified_flag:
            model = PolynomialModel(degree=2,TRUE_BETA = torch.tensor([1.0, 2.0, -0.5]))
            label = method +"_well"
        else:
            true_beta = torch.tensor([1.0, 2.0, -0.5,-5.0,6,9])[:misspecified_flag+1]
            model = PolynomialModel(degree=1,TRUE_BETA = true_beta,BEST_BETA = true_beta)
            label = method +"_mis_" + str(misspecified_flag)

        experiment = BOEDExperiment(model,num_candidates=60,eig_method= eig_method, dir = dir)


        posterior_mean = model.prior_mean
        posterior_sd = model.prior_sd
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed+ ex_idx*10)
            torch.manual_seed(seed+ ex_idx*10)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)
            torch.manual_seed(seed)


        xis = torch.tensor([])  # Design points
        ys = torch.tensor([])   # Observations
        if method[:3] == "dad":
            y_outcomes = []
            x_designs = []

        generalization_error[ex_idx,0], mean_bias2, mean_variance, mean_cross = test_phase(model, posterior_mean, posterior_sd, test_x, xis,ys, axes[0],0,misspecified_flag,PLOT)
        covarite_shift_degrees[ex_idx,0] = 0

        run_df = pd.DataFrame(torch.tensor([0.00000]).numpy())
        run_df.columns = ["xi"]
        run_df["ex_idx"] = ex_idx
        run_df["round_idx"] = 0
        run_df["observations"] = 0
        run_df["covariate_shift_degree"] = covarite_shift_degrees[ex_idx,0]
        run_df["generalization_error"] = generalization_error[ex_idx,0]
        run_df["posterior_mean"] = [posterior_mean]
        run_df["posterior_sd"] = [posterior_sd]
        run_df["mean_bias2"] = mean_bias2, 
        run_df["mean_variance"] = mean_variance,
        run_df["mean_cross"] = mean_cross
        output.append(run_df)
        
        if method == "random":
                # best_x = (torch.round(torch.rand((1)) * (8) / 0.1) * 0.1 - 4)
            best_x_box = torch.empty(num_rounds+1).uniform_(-4, 4)# np.ran

        for round_idx in range(1,num_rounds+1):
            # Optimize design
            if method == "boed":
                best_x,_ = experiment.optimize_design(xis, ys, model)
            elif method.startswith("boed_lambda_"):
                best_x,_ = experiment.optimize_design(xis, ys, model, float(method[len("boed_lambda_"):]))
            elif method == "initial":
                if round_idx ==1:
                    best_x = torch.tensor([0.0])
                else:
                    best_x,_ = experiment.optimize_design(xis, ys, model)

            elif method == "random":
                best_x = (torch.round(torch.rand((1)) * (8) / 0.1) * 0.1 - 4)
                # best_x = torch.empty(1).uniform_(-4, 4)# np.random.uniform(0,1)
                # best_x = best_x_box[round_idx-1].unsqueeze(-1)

            elif method[:3] == "dad":
                model_location = f"./mlruns/{experiment_id}/{run_id}/artifacts/model"
                ho_model = mlflow.pytorch.load_model(model_location)
                with torch.no_grad():
                    best_x = compute_design(f"xi{round_idx}", ho_model.design_net.lazy(*zip(x_designs, y_outcomes)))
                    best_x = torch.clamp(best_x, -4, 4)
                
            else:
                print(f"the method is: {method}")
                print("check the method again!")
                exit()
            print("===============" + str(round_idx))
            print("best_x",best_x)
            
            # Generate synthetic observation
            y = model.synthetic_data(best_x)
            if method[:3] == "dad":
                best_x = best_x.unsqueeze(-1)
                y = y.unsqueeze(-1)
                y_outcomes.append(y)
                x_designs.append(best_x)
            xis = torch.cat([xis, best_x])
            ys = torch.cat([ys, y])


            # Infer posterior
            posterior_mean, posterior_sd = experiment.run_mcmc(xis, ys)

            
            generalization_error[ex_idx,round_idx], mean_bias2, mean_variance, mean_cross = test_phase(model, posterior_mean, posterior_sd, test_x, xis,ys, axes[round_idx],round_idx,misspecified_flag, PLOT)
            covarite_shift_degrees[ex_idx,round_idx] = compute_covarite_shift(xis,covarite_shift_method = covarite_shift_method)
            # update the para of model
            model.prior_mean = posterior_mean
            model.prior_sd = posterior_sd

            run_df = pd.DataFrame(best_x.numpy())
            run_df.columns = ["xi"]
            run_df["ex_idx"] = ex_idx
            run_df["round_idx"] = round_idx
            run_df["observations"] = y
            run_df["covariate_shift_degree"] = covarite_shift_degrees[ex_idx,round_idx]
            run_df["generalization_error"] = generalization_error[ex_idx,round_idx]
            run_df["posterior_mean"] = [posterior_mean]
            run_df["posterior_sd"] = [posterior_sd]
            run_df["mean_bias2"] = mean_bias2, 
            run_df["mean_variance"] = mean_variance,
            run_df["mean_cross"] = mean_cross
            output.append(run_df)
        
            
        torch.cuda.empty_cache()

        # delte
        if num_ex != 1:
            del model, experiment
            del xis, ys
            del posterior_mean, posterior_sd
            gc.collect()  
    save_path = dir + label + '.csv'
    res = pd.concat(output)
    res.to_csv(save_path, index=False)
    print("res save to", save_path)
            

    return generalization_error,rmse_values, label, covarite_shift_degrees



if __name__ == "__main__":
    # Initialize model and experiment
    # seed 42, 2023, 2024, 777
 
    # orginal compare 
    # need to add the trained dad network in the for loop, about the line 275.
    method_box = ["random","random","boed","boed","dad","dad"]
    misspecified_flag_box = [False,2,False,2,False,2]
    output_dir = "../../results/test05/result42_final/"




    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(method_box) != len(misspecified_flag_box):
        print("check len of box!!!!")
        exit()
    # pyro.set_rng_seed(44)
    generalization_errors=[]
    covarite_shift_degrees = []
    rmse_values_miss = []
    labels = []

    for i, (method, misspecified_flag) in enumerate(zip(method_box, misspecified_flag_box)):
        print("============================")
        print(method+ str(misspecified_flag))
        if method == "dad":
            if not misspecified_flag:
                experiment_id =""
                run_id = ""  # the wellspecified model directory
            else: # misspecified = True
                experiment_id =""
                run_id = "" # the misspecified model directory; 
                
        else:
            experiment_id =""
            run_id = ""
            exit()
        generalization_error, rmse_values_mis, label, covarite_shift_degree= experiment(method = method, 
                                                                 misspecified_flag = misspecified_flag, eig_method = "svi_eig", 
                                                                 num_rounds=10,
                                                                 num_ex = 20,
                                                                 experiment_id= experiment_id,
                                                                 run_id = run_id,
                                                                 dir=output_dir,covarite_shift_method = "mmd",
                                                                 seed=42)
        print("generalization_error",generalization_error)
        generalization_errors.append(generalization_error)
        covarite_shift_degrees.append(covarite_shift_degree)
        rmse_values_miss.append(rmse_values_mis)
        labels.append(label)

    plot_generalization_error_vs_round(output_dir,generalization_errors,labels)
    plot_covariet_shift_vs_round(output_dir,covarite_shift_degrees, labels)
