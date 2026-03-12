import numpy as np
from pricing_engine import QMC_fourier_pricing_engine, NF_QMC_fourier_pricing_engine
from models import covariance_matrix
import torch
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as colr
#import sklaern
from sklearn.datasets import make_spd_matrix
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")
"""
Basket Put MC Benchmark:
MC price estimate = 7.49067 
MC relative statistical error =  0.0003
Call on min MC Benchmark:
MC price estimate = 4.57777 
MC relative statistical error =  0.0004
"""

def data_analysis():
    basket_put_price = 7.49067 
    call_on_min_price = 4.57777 
    inputs = pd.read_csv("input_nfm_parameter_space.csv")
    outputs = pd.read_csv("checkpoint.csv")
    fulldf = pd.concat((inputs,outputs),axis = 1).drop(columns=("Unnamed: 0"))
    fulldf.to_csv(".\\Visualization\\CurrentResults.csv")
    fulldf = fulldf.dropna()
    fulldf["nfm_losses"] = [float(elem.strip("tensor(").rstrip(", grad_fn=<SubBackward0>)").rstrip(", grad_fn=<NegBackward0>)"))for elem in fulldf["nfm_losses"]]
    print(fulldf["nfm_losses"])
    fulldf["nfm_pricing_deviation"] = np.abs(1-fulldf["price_estimates_NF"]/basket_put_price)
    fulldf["qmc_pricing_deviation"] = np.abs(1-fulldf["price_estimates_classic"]/basket_put_price)
    fulldf.to_csv(".\\Visualization\\test.csv")
    backward_df = fulldf[fulldf["forward_kls"]==False]
    cmap = plt.get_cmap("tab10")
    rgb_list = [cmap(i)[:3] for i in range(6)]
    fig,ax = plt.subplots(figsize=(10, 5))
    for blocks in [8]:
        for i, num_samples in enumerate([*np.array(np.logspace(7,12,6,base=2.0)).astype("int")]):
            for j, optimizer in enumerate(["sgd","adam"]):
                currdf = backward_df[backward_df["num_samples_init"]==num_samples][backward_df["optimizer_methods"]==optimizer]
                print(rgb_list[i])
                if j:
                    m = "x"
                    ax.scatter(currdf["nfm_losses"],currdf["nfm_pricing_deviation"],label = f"{num_samples} Training Samples",color=rgb_list[i],marker = m)
                else:
                    m="o"
                    ax.scatter(currdf["nfm_losses"],currdf["nfm_pricing_deviation"],color=rgb_list[i],marker = m)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Normalizing Flow Training Losses [-]")
        plt.ylabel("Relative Pricing Difference [-]")
        plt.plot(np.linspace(min(backward_df["nfm_losses"]),max(backward_df["nfm_losses"]),len(backward_df["nfm_losses"])),backward_df["qmc_pricing_deviation"],linestyle = "--",label = "QMC Pricing Deviation")
        plt.legend()
        plt.title("Relative Pricing Difference Comparison QMC, NFM for 2**14 Samples and 30 Shifts")
        fig.savefig(f".\\Visualization\\pricing_loss_backward.png")
        plt.close()
    cmap = plt.get_cmap("tab10")
    rgb_list = [cmap(i)[:3] for i in range(6)]
    fig,ax = plt.subplots(figsize=(10, 5))
    for blocks in [8]:
        for i, num_samples in enumerate([*np.array(np.logspace(7,12,6,base=2.0)).astype("int")]):
            for j, optimizer in enumerate(["sgd","adam"]):
                currdf = backward_df[backward_df["num_samples_init"]==num_samples][backward_df["optimizer_methods"]==optimizer]
                print(rgb_list[i])
                if j:
                    m = "x"
                    ax.scatter(currdf["nfm_losses"],currdf["error_estimates_NF"],color=rgb_list[i],marker = m)
                else:
                    m="o"
                    ax.scatter(currdf["nfm_losses"],currdf["error_estimates_NF"],label = f"{num_samples} Training Samples",color=rgb_list[i],marker = m)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Normalizing Flow Training Losses [-]")
        plt.ylabel("Relative Pricing Difference [-]")
        plt.plot(np.linspace(min(backward_df["nfm_losses"]),max(backward_df["nfm_losses"]),len(backward_df["nfm_losses"])),backward_df["error_estimates_classic"],linestyle = "--",label = "QMC Pricing Deviation")
        plt.legend()
        plt.title("Pricing Deviation Comparison QMC, NFM for 2**14 Samples and 30 Shifts")
        fig.savefig(f".\\Visualization\\pricing_deviation_backward.png")
    forward_df = fulldf[fulldf["forward_kls"]==True]
    cmap = plt.get_cmap("tab10")
    rgb_list = [cmap(i)[:3] for i in range(6)]
    fig,ax = plt.subplots(figsize=(10, 5))
    for blocks in [8]:
        for i, num_samples in enumerate([*np.array(np.logspace(7,12,6,base=2.0)).astype("int")]):
            for j, optimizer in enumerate(["sgd","adam"]):
                currdf = forward_df[forward_df["num_samples_init"]==num_samples][forward_df["optimizer_methods"]==optimizer]
                print(rgb_list[i])
                if j:
                    m = "x"
                    ax.scatter(currdf["nfm_losses"],currdf["nfm_pricing_deviation"],label = f"{num_samples} Training Samples",color=rgb_list[i],marker = m)
                else:
                    m="o"
                    ax.scatter(currdf["nfm_losses"],currdf["nfm_pricing_deviation"],color=rgb_list[i],marker = m)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Normalizing Flow Training Losses [-]")
        plt.ylabel("Relative Pricing Difference [-]")
        plt.plot(np.linspace(min(forward_df["nfm_losses"]),max(forward_df["nfm_losses"]),len(forward_df["nfm_losses"])),forward_df["qmc_pricing_deviation"],linestyle = "--",label = "QMC Pricing Deviation")
        plt.legend()
        plt.title("Relative Pricing Difference Comparison QMC, NFM for 2**14 Samples and 30 Shifts")
        fig.savefig(f".\\Visualization\\pricing_loss_forward.png")
        plt.close()
    cmap = plt.get_cmap("tab10")
    rgb_list = [cmap(i)[:3] for i in range(6)]
    fig,ax = plt.subplots(figsize=(10, 5))
    for blocks in [8]:
        for i, num_samples in enumerate([*np.array(np.logspace(7,12,6,base=2.0)).astype("int")]):
            for j, optimizer in enumerate(["sgd","adam"]):
                currdf = forward_df[forward_df["num_samples_init"]==num_samples][forward_df["optimizer_methods"]==optimizer]
                print(rgb_list[i]) 
                if j:
                    m = "x"
                    ax.scatter(currdf["nfm_losses"],currdf["error_estimates_NF"],color=rgb_list[i],marker = m)
                else:
                    m="o"
                    ax.scatter(currdf["nfm_losses"],currdf["error_estimates_NF"],label = f"{num_samples} Training Samples",color=rgb_list[i],marker = m)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Normalizing Flow Training Losses [-]")
        plt.ylabel("Relative Pricing Difference [-]")
        plt.plot(np.linspace(min(forward_df["nfm_losses"]),max(forward_df["nfm_losses"]),len(forward_df["nfm_losses"])),forward_df["error_estimates_classic"],linestyle = "--",label = "QMC Pricing Deviation")
        plt.legend()
        plt.title("Pricing Deviation Comparison QMC, NFM for 2**14 Samples and 30 Shifts")
        fig.savefig(f".\\Visualization\\pricing_deviation_forward.png")
    print(backward_df["num_samples_init"])

    full_pricing_sweep_df = pd.read_csv(".\\Visualization//Analysis2D.csv").drop(columns=("Unnamed: 0"))
    full_pricing_sweep_df = full_pricing_sweep_df[full_pricing_sweep_df["forward_kls"]==False]
    basket_put_price_NF = np.mean(full_pricing_sweep_df[f"price_estimates_NF_{2**14}"])
    basket_put_price_QMC = 0.5*(np.mean(full_pricing_sweep_df[f"price_estimates_QMC_{2**14}"])+np.mean(full_pricing_sweep_df[f"price_estimates_QMC_{2**13}"]))
    print(full_pricing_sweep_df)
    num_samples_pricing_list = np.logspace(4,14,11,base = 2, dtype = int)
    num_samples_trainig_list = np.logspace(10,12,3,base = 2, dtype = int)
    for num_pricing_samples in num_samples_pricing_list:
        full_pricing_sweep_df[f"price_estimate_difference_NF_{num_pricing_samples}"] = np.abs(1-basket_put_price_NF/full_pricing_sweep_df[f"price_estimates_NF_{num_pricing_samples}"])
        full_pricing_sweep_df[f"price_estimate_difference_QMC_{num_pricing_samples}"] = np.abs(1-basket_put_price_NF/full_pricing_sweep_df[f"price_estimates_QMC_{num_pricing_samples}"])
    adam_df = full_pricing_sweep_df[full_pricing_sweep_df["optimizer_methods"]=="adam"]
    sgd_df = full_pricing_sweep_df[full_pricing_sweep_df["optimizer_methods"]=="sgd"]
    cmap = plt.get_cmap("tab10")
    rgb_list = [cmap(i)[:3] for i in range(6)]
    fig,ax = plt.subplots(figsize=(10, 5))
    for i, num_samples_training in enumerate(num_samples_trainig_list):
        currdf_adam = adam_df[adam_df["num_samples_init"]==num_samples_training]
        currdf_sgd = sgd_df[sgd_df["num_samples_init"]==num_samples_training]
        ax.scatter(currdf_adam["nfm_losses"],currdf_adam[f"error_estimates_NF_{1024}"]/currdf_adam[f"price_estimates_NF_{1024}"],color=rgb_list[i],marker = "o", label = f"{num_samples_training} Training Samples, Adam")
        ax.scatter(currdf_sgd["nfm_losses"],currdf_sgd[f"error_estimates_NF_{1024}"]/currdf_sgd[f"price_estimates_NF_{1024}"],color=rgb_list[i],marker = "x", label = f"{num_samples_training} Training Samples, SGD")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Normalizing Flow Training Losses [-]")
    plt.ylabel("Relative Pricing Difference [-]")
    plt.plot(np.linspace(min(full_pricing_sweep_df["nfm_losses"]),max(full_pricing_sweep_df["nfm_losses"]),len(full_pricing_sweep_df["nfm_losses"])),full_pricing_sweep_df["price_estimate_difference_QMC_1024"],linestyle = "--",label = "QMC Pricing Deviation")
    plt.legend()
    plt.title("Relative Pricing Difference Comparison QMC, NFM for 2**10 Samples and 30 Shifts")
    fig.savefig(f".\\Visualization\\Relative_Pricing_deviation_training_2D.png")
    plt.close()
    fig,ax = plt.subplots(figsize=(10, 5))
    adam_df_4096 = adam_df[adam_df["num_samples_init"] == 4096]
    ax.plot(num_samples_pricing_list,
            [np.mean(adam_df[f"price_estimate_difference_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference NF Adam")
    ax.plot(num_samples_pricing_list,
            [np.mean(sgd_df[f"price_estimate_difference_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference NF SGD")
    ax.plot(num_samples_pricing_list,
            [np.mean(adam_df[f"price_estimate_difference_QMC_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference QMC")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of Pricing Samples [-]")
    plt.ylabel("Relative Pricing Difference [-]")
    plt.title("Relative Pricing Difference Comparison QMC, NFM for 2**12 Training Samples and High LR")
    fig.savefig(f".\\Visualization\\Relative_Pricing_diff_sweep_2D.png")
    plt.close()
    fig,ax = plt.subplots(figsize=(10, 5))
    adam_df_4096 = adam_df[adam_df["num_samples_init"] == 4096]
    ax.plot(num_samples_pricing_list,
            [np.mean(adam_df[f"error_estimates_NF_{num_samples_pricing}"])/np.mean(adam_df[f"price_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference NF Adam")
    ax.plot(num_samples_pricing_list,
            [np.mean(sgd_df[f"error_estimates_NF_{num_samples_pricing}"])/np.mean(sgd_df[f"price_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference NF SGD")
    ax.plot(num_samples_pricing_list,
            [np.mean(adam_df[f"error_estimates_QMC_{num_samples_pricing}"])/np.mean(sgd_df[f"price_estimates_QMC_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list],
            label = "Pricing Difference QMC")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of Pricing Samples [-]")
    plt.ylabel("Relative Pricing Deviation [-]")
    plt.title("Pricing Deviation Comparison Analytical, NFM averaged over Design Space")
    fig.savefig(f".\\Visualization\\Relative_Pricing_deviation_sweep_2D.png")
    plt.close()
    print("QMC:")
    print([np.mean(adam_df[f"error_estimates_QMC_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    print([np.mean(adam_df[f"price_estimates_QMC_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    print("Normalizing Flow, Adam:")
    print([np.mean(adam_df[f"error_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    print([np.mean(adam_df[f"price_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    print("Normalizing Flow, SGD:")
    print([np.mean(sgd_df[f"error_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    print([np.mean(sgd_df[f"price_estimates_NF_{num_samples_pricing}"])for num_samples_pricing in num_samples_pricing_list])
    #print(fulldf)

def main():
    model_name = 'VG'
    payoff_name = 'basket_put'
    basket_put_price = 7.49067 
    call_on_min_price = 4.57777 
    payoff_data = {"basket_put":basket_put_price,"call_on_min":call_on_min_price}

    num_layers_list = [16]
    max_iters = [500,1000,2000]
    num_samples_init = [*np.array(np.logspace(8,10,3,base=2.0)).astype("int")]
    jump_iters = [100000]
    forward_kls = [False, True]
    optimizer_methods = ["adam","sgd"]
    lrs = [1e-02]
    weight_decays = [1e-06]

    transform_parameter_array = np.array([*product(num_layers_list,max_iters,num_samples_init,jump_iters,forward_kls,optimizer_methods,lrs,weight_decays)])
    transfrom_parameter_list = [*product(num_layers_list,max_iters,num_samples_init,jump_iters,forward_kls,optimizer_methods,lrs,weight_decays)]
    print(transform_parameter_array[0])
    transform_parameter_df = pd.DataFrame({
        "num_layers_list":transform_parameter_array[:,0].astype(np.float64).astype(np.int64),
        "max_iters":transform_parameter_array[:,1].astype(np.float64).astype(np.int64),
        "num_samples_init":transform_parameter_array[:,2].astype(np.float64).astype(np.int64),
        "jump_iters":transform_parameter_array[:,3].astype(np.float64).astype(np.int64),
        "forward_kls":[True if elem == "True" else False for elem in transform_parameter_array[:,4]],
        "optimizer_methods":transform_parameter_array[:,5].astype(np.str_),
        "lrs":transform_parameter_array[:,6].astype(np.float64),
        "weight_decays":transform_parameter_array[:,7].astype(np.float64)
    })
    transform_parameter_df.to_csv(".\\Visualization\\input_nfm_parameter_space_2D_test.csv")
    #    def assignParamRows(self):      #Assembles the given Parameter ranges to all possible combinations
    #
    #    self.RowArray=np.array([*product(*self.params)])
    #    self.DataFrame=pd.DataFrame({key:self.RowArray.T[i] for i,key in enumerate(self.description)})
    #parameter_df = pd.DataFrame({
    #    "payoff_name": ["basket_put","call_on_min"],
    #    ""
    #})
    # Define parameters (ρ_ij = 0.2 / (1 + 0.1|i−j|))
    #transform_params = num_layers, max_iter, num_samples_init, jump_iter, forward_kl, optimizer_method,lr, weight_decay
    d = 2
    S0 = 100 * np.ones(d)
    K = 100
    r = 0
    q = 0
    T = 1
    nu = 0.1
    sigma = 0.4 * np.ones(d)
    theta = -0.3 * np.ones(d)
    rho = 0.2*np.ones((d,d))
    rho = np.array([[elem/(1+0.1*np.abs(i-j))for j,elem in enumerate(row)]for i,row in enumerate(rho)])
    #SIGMA = make_spd_matrix(n_dim = d)
    SIGMA = covariance_matrix(sigma,rho)
    print(f"Covariance Matrix:\n{SIGMA}")
    TOLR = None # specify relative error
    VG_option_params = (d,S0,K,r,q,T,SIGMA,theta,nu)
    N_samples_list = [*np.logspace(4,8,5,base = 2,dtype=int)]
    S_shifts = 30
    output_df= pd.DataFrame({
        "nfm_losses":[],
        "N_samples":[],
        **{f"price_estimates_NF_{N_samples}":[]for N_samples in N_samples_list},
        **{f"error_estimates_NF_{N_samples}":[]for N_samples in N_samples_list},
        **{f"price_estimates_QMC_{N_samples}":[]for N_samples in N_samples_list},
        **{f"error_estimates_QMC_{N_samples}":[]for N_samples in N_samples_list},
    })
    nfm_losses = []
    price_estimates_NF = len(N_samples_list)*[[]]
    error_estimates_NF = len(N_samples_list)*[[]]
    price_estimates_classic = len(N_samples_list)*[[]]
    error_estimates_classic = len(N_samples_list)*[[]]
    price_estimates_classic_source = len(N_samples_list)*[[]]
    error_estimates_classic_source = len(N_samples_list)*[[]]
    print(error_estimates_classic)
    N_Samples_curr_list=[]
    for i, N_samples in enumerate(N_samples_list):
        print(i)
        price_estimates_classic_source[i], error_estimates_classic_source[i] = QMC_fourier_pricing_engine(
            model_name,
            payoff_name,
            VG_option_params,
            N_samples,
            S_shifts,
            transform_distribution= None,
            transform_params= None,
            TOLR = TOLR
        )
    for j, row in enumerate(transfrom_parameter_list):
        for i, N_samples in enumerate(N_samples_list):
            if i:
                (price_estimate_NF, error_estimate_NF), nfm_loss, used_model = NF_QMC_fourier_pricing_engine(
                    model_name,
                    payoff_name,
                    VG_option_params,
                    N_samples,
                    S_shifts,
                    transform_distribution= None,
                    transform_params= row,
                    TOLR = TOLR,
                    nfm_model=used_model
                )
            else:
                (price_estimate_NF, error_estimate_NF), nfm_loss, used_model = NF_QMC_fourier_pricing_engine(
                    model_name,
                    payoff_name,
                    VG_option_params,
                    N_samples,
                    S_shifts,
                    transform_distribution= None,
                    transform_params= row,
                    TOLR = TOLR
                )
            if not i:
                nfm_losses.append(nfm_loss)
            price_estimates_NF[i] = [*price_estimates_NF[i]]+[price_estimate_NF]
            error_estimates_NF[i] = [*error_estimates_NF[i]]+[error_estimate_NF]
            price_estimates_classic[i] = [*price_estimates_classic[i]]+[price_estimates_classic_source[i]]
            error_estimates_classic[i] = [*error_estimates_classic[i]]+[error_estimates_classic_source[i]]
            #print(price_estimates_NF)
        print(i,j)
        elemstring = '-'.join([f"{name}={str(value)}"for name, value in zip(transform_parameter_df,row)])
        print(elemstring)
        plot_logprob(f"sampling_space_{elemstring}",used_model)
        output_df= pd.DataFrame({
            "nfm_losses":nfm_losses,
            **{f"price_estimates_NF_{N_samples}":price_estimates_NF[k]for k,N_samples in enumerate(N_samples_list)},
            **{f"error_estimates_NF_{N_samples}":error_estimates_NF[k]for k,N_samples in enumerate(N_samples_list)},
            **{f"price_estimates_QMC_{N_samples}":price_estimates_classic[k]for k,N_samples in enumerate(N_samples_list)},
            **{f"error_estimates_QMC_{N_samples}":error_estimates_classic[k]for k,N_samples in enumerate(N_samples_list)}
        })
        output_df.to_csv(".\\Visualization\\checkpoint2D_test.csv")

    full_df = pd.concat((transform_parameter_df,output_df),axis ="columns")
    full_df.to_csv(".\\Visualization\\Analysis2D_test.csv")

    #print(f"Estimated Price: {round(price_estimate,5)}, Statistical Error: {round(error_estimate,5)}, Relative Error: {round(error_estimate / price_estimate,5)}")
    #print("Normalizing Flow based Sampling:")
    #print(f"Estimated Price: {round(price_estimate_NF,5)}, Statistical Error: {round(error_estimate_NF,5)}, Relative Error: {round(error_estimate_NF / price_estimate_NF,5)}")


def plot_logprob(filename, nfm, gridsize = 2**6):
    grid_x,grid_y = torch.meshgrid(torch.linspace(2**(-5),1-2**(-5),gridsize),torch.linspace(2**(-5),1-2**(-5),gridsize),indexing = "ij")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    print(points.shape)
    with torch.no_grad():
        distribution_points,base_log_prob = nfm.q0.forward_given_samples(points)
        print(distribution_points)
        _, data = nfm.forward_and_log_det(distribution_points)
        data += base_log_prob
    Z = data.reshape(grid_x.shape)
    plt.pcolormesh(grid_x,grid_y,Z, cmap='autumn')
    plt.title("Log Probability of [0,1]x[0,1] grid passed through the normalizing flow")
    plt.colorbar(label='Log Probability Density')
    plt.savefig(f".//Visualization//Log_Prob_{filename}.png")
    plt.close()
    plt.pcolormesh(grid_x,grid_y,torch.exp(Z), cmap='autumn')
    plt.title("Probability of [0,1]x[0,1] grid passed through the normalizing flow")
    plt.colorbar(label='Probability Density')
    plt.savefig(f".//Visualization//Prob_{filename}.png")
    plt.close()

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    SCIPY_ARRAY_API=1
    #plot_logprob("test", None)
    main()
    data_analysis()