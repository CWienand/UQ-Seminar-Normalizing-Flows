import numpy as np
import scipy.linalg as la
from scipy.stats import norm
from models import NIGModel, GHModel, VGModel, GBMModel
from payoffs import CallOnMin, PutOnMax, BasketPut, SpreadCall, CashOrNothingPut
from domain_transformation import MLTransform, MSTransform, MNTransform
from damping_optimization import optimize_damping_parameters
import qmcpy
import torch
import normflows as nf
import nfm_distributions
import matplotlib.pyplot as plt
from tqdm import tqdm
from MLP import CustomMLP

class PricingEngine:
    def __init__(self, model, payoff, transform):
        self.model = model
        self.payoff = payoff
        self.transform = transform
    
    def optimize_R(self, d, S0, K):
        """
        Wrapper method that calls the external function to optimize R.
        """
        return optimize_damping_parameters(self.payoff, self.model, d, S0, K)

    
    def compute_price(self, d, S0, K, r, T, N_samples, S_shifts):
    # Optimize R based on the model and payoff constraints
    
        R = self.optimize_R(d = d, S0=S0, K=K)
        print("R = ", R)
        
        # Calculate X0 and payoff scaling factor based on the specific payoff
        X0 = self.payoff.calculate_X0(S0, K)
        payoff_scaling = self.payoff.scaling(K)
        
        # Calculate the constant factor for the pricing formula
        d = len(S0)
        constant_factor = payoff_scaling * (2 * np.pi)**(-d) * np.exp(-r * T) * np.exp(-R @ X0)
        
        # Decompose the transformation matrix for QMC sampling
        L_IS = self.transform.decompose()
        
        # Initialize array to store QMC replicates
        RQMC_replicates = np.zeros(S_shifts)
    
        for s in range(S_shifts):
            # Generate QMC points and map them using ICDF
            QMC_points = qmcpy.DigitalNetB2(d + 1, graycode=True, randomize='DS', seed=s).gen_samples(N_samples)
            QMC_points_last = QMC_points[:, -1]
            QMC_points_mapped = norm.ppf(q=QMC_points[:, :-1], loc=0, scale=1)
            
            QMC_estimate = 0
            for n in range(N_samples):
                # Transform the QMC points using L_IS and ICDF
                u = L_IS @ QMC_points_mapped[n] * self.transform.icdf(QMC_points_last[n])
                
                # Calculate the characteristic function and payoff transform
                Phi = self.model.characteristic_function(u + 1j*R)
                Phat = self.payoff.fourier_transform(u + 1j*R)
                
                # Calculate the PDF of the transformed distribution
                psi = self.transform.pdf(u)
                
                # Accumulate the estimate for this sample
                QMC_estimate += constant_factor * np.real(np.exp(1j * u @ X0) * Phi * Phat) / psi
            
            # Store the replicate result
            RQMC_replicates[s] = QMC_estimate / N_samples
    
        # Calculate the final estimate and statistical error
        RQMC_estimate = np.mean(RQMC_replicates)
        RQMC_stat_error = 1.96 * np.std(RQMC_replicates) / np.sqrt(S_shifts)
        return RQMC_estimate, RQMC_stat_error


class NF_PricingEngine:
    def __init__(self, model, payoff, normalizing_flow):
        self.model = model
        self.payoff = payoff
        self.normalizing_flow = normalizing_flow
    
    def optimize_R(self, d, S0, K):
        """
        Wrapper method that calls the external function to optimize R.
        """
        return optimize_damping_parameters(self.payoff, self.model, d, S0, K)
    
    def compute_price(self, d, S0, K, r, T, N_samples, S_shifts):
    # Optimize R based on the model and payoff constraints
    
        R = self.optimize_R(d = d, S0=S0, K=K)
        print("R = ", R)
        
        # Calculate X0 and payoff scaling factor based on the specific payoff
        X0 = self.payoff.calculate_X0(S0, K)
        payoff_scaling = self.payoff.scaling(K)
        
        # Calculate the constant factor for the pricing formula
        d = len(S0)
        constant_factor = payoff_scaling * (2 * np.pi)**(-d) * np.exp(-r * T) * np.exp(-R @ X0)
        
        # Initialize array to store QMC replicates
        RQMC_replicates = np.zeros(S_shifts)
    
        for s in range(S_shifts):
            # Transform the QMC points using L_IS and ICDF
            u, logpsi = self.normalizing_flow.sample(N_samples)
            u = u.detach().numpy()
            # Calculate the PDF of the transformed distribution
            psi = torch.exp(logpsi).detach().numpy()
            QMC_estimate = 0
            for n in range(N_samples):

                # Calculate the characteristic function and payoff transform
                Phi = self.model.characteristic_function(u[n] + 1j*R)
                Phat = self.payoff.fourier_transform(u[n] + 1j*R)
                
                # Accumulate the estimate for this sample
                QMC_estimate += constant_factor * np.real(np.exp(1j * u[n] @ X0) * Phi * Phat) / psi[n]
                
                # Store the replicate result
                RQMC_replicates[s] = QMC_estimate / N_samples
    
        # Calculate the final estimate and statistical error
        RQMC_estimate = np.mean(RQMC_replicates)
        RQMC_stat_error = 1.96 * np.std(RQMC_replicates) / np.sqrt(S_shifts)
        return RQMC_estimate, RQMC_stat_error
        

def assign_flows(K=16,latent_size = 4,non_lin = "tanh"):
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    for i in range(K):
        s = CustomMLP([latent_size,32*latent_size,16*latent_size, latent_size], init_zeros=True)
        t = CustomMLP([latent_size,32*latent_size,16*latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]
        #flows += [nf.flows.Permute(latent_size,mode = "swap")]
    return flows


def train_flow(
        flow,max_iter = 1000, num_samples_init = 2**7, jump_iter = 10000, annealing_goal = None,
        forward_kl = False, optimizer_method = "Adam",lr = 1e-02, weight_decay = 1e-06
        ):
    num_samples = num_samples_init
    loss_hist = np.array([])
    if forward_kl:
        max_iter*=2
        num_samples_init*=8
    if optimizer_method.lower() == "adam":
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr,weight_decay=weight_decay)
    if optimizer_method.lower() == "sgd":
        optimizer = torch.optim.SGD(flow.parameters(), lr=lr,weight_decay=weight_decay)
    if annealing_goal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_iter,annealing_goal)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        if not it%jump_iter:
            #num_samples*=2
            print("doubled")
        if forward_kl:
            X = flow.p.sample(num_samples)
            loss = flow.forward_kld(X)
        else:
            loss = flow.reverse_kld(num_samples)
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            if annealing_goal:
                scheduler.step()
        else:
            optimizer = torch.optim.SGD(flow.parameters(), lr=1e-6)
            #scheduler.step(epoch = it)
            print("Loss untypical")
        
    plt.figure(figsize=(20, 10))
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(np.abs(loss_hist), label='KL Divergence')
    plt.legend()
    plt.savefig(f"loss_nfm.png",dpi = 150)
    plt.close()
    if forward_kl:
        X = flow.p.sample(2**13)
        return flow,flow.forward_kld(X)
    else:
        return flow,flow.reverse_kld(2**13)


def NF_QMC_fourier_pricing_engine(
        model_name, payoff_name, option_params, N_samples, S_shifts, 
        transform_distribution=None, transform_params=None, weights=None, TOLR = None, nfm_model = None):
    
    # Initialize model
    if model_name == 'NIG':
        d, S0, K, r, T, q, alpha, beta, delta, DELTA = option_params
        model = NIGModel(d, T, r, q, alpha, beta, delta, DELTA)
        if transform_distribution is None:
            transform_distribution = 'ML'
            transform_params = 2 / (delta**2 * T**2) * la.inv(DELTA)
        
    elif model_name == 'GH':
        d,S0,K,r,T,q,alpha,beta,delta,DELTA,lambda_var = option_params
        model = GHModel(d, T, r, q, alpha, beta, delta, DELTA, lambda_var)
        if transform_distribution is None:
            transform_distribution = 'ML'
            transform_params = 2 / (delta**2 * T**2) * la.inv(DELTA)
        
    elif model_name == 'VG':
        d,S0,K,r,q,T,SIGMA,theta,nu = option_params
        model = VGModel(d, T, r, q, SIGMA, theta, nu)
        location = np.zeros_like(np.diagonal(la.inv(SIGMA)))
        if nfm_model:
            trained_nfm = nfm_model
            achieved_loss = nfm_model.reverse_kld(2**13)
        else:
            base = nfm_distributions.Multivariate_Diag_t_qmc(loca = location,diag = np.diagonal(la.inv(SIGMA)),df = 2*T / nu - d)
            target = nfm_distributions.Multivariate_t_target(loc = location,cov = la.inv(SIGMA),df = 2*T / nu - d)
            if transform_params is not None:
                num_layers, max_iter, num_samples_init, jump_iter, \
                forward_kl, optimizer_method,lr, weight_decay = transform_params
                flows = assign_flows(K=num_layers,latent_size=len(location),non_lin="tanh")
                nfm = nf.NormalizingFlow(q0 = base, flows=flows, p=target)
                trained_nfm,achieved_loss = train_flow(
                    nfm,max_iter=max_iter, num_samples_init=num_samples_init, 
                    jump_iter=jump_iter,forward_kl=forward_kl, 
                    optimizer_method=optimizer_method,
                    lr=lr, weight_decay=weight_decay
                    )
            else:
                flows = assign_flows(latent_size=len(location),non_lin="tanh")
                nfm = nf.NormalizingFlow(q0 = base, flows=flows, p=target)
                trained_nfm,achieved_loss = train_flow(nfm)
        
            
    elif model_name == 'GBM':
        d,S0,K,r,T,q,SIGMA = option_params
        model = GBMModel(d, T, r, q, SIGMA)
    
    # Initialize payoff
    if payoff_name == 'call_on_min':
        payoff = CallOnMin()
    
    elif payoff_name == 'put_on_max':
        payoff = PutOnMax()
        
    elif payoff_name == 'spread_call':
        payoff = SpreadCall()
   
    elif payoff_name == 'basket_put':
        if weights is None:
            weights = (1/d) * np.ones(d)  # Default to equi-weighted option
        payoff = BasketPut(weights=weights)
    
    elif payoff_name == 'cash_or_nothing_put':
        payoff = CashOrNothingPut()
    

    # Create pricing engine
    engine = NF_PricingEngine(model=model, payoff=payoff, normalizing_flow=trained_nfm)
    
    
    if S_shifts < 30:
        print("Warning: The statistical error estimation of RQMC is not reliable because S_shifts < 30.")
    # Compute price
    if (TOLR is not None): # if the user specifies targer relative error
        rel_stat_error = 2 * TOLR # Initialize relative statistical error
        N_samples = 2
        while (rel_stat_error > TOLR):
            N_samples = 2 * N_samples # double the number of samples if relative tolerance not achieved.
            price, stat_error = engine.compute_price(d=d,S0=S0,K=K,r=r,T=T,N_samples=N_samples,S_shifts=S_shifts)
            rel_stat_error = stat_error / price
        return (price, stat_error), achieved_loss.detach().numpy(), trained_nfm
    else:
        return engine.compute_price(d=d,S0=S0,K=K,r=r,T=T,N_samples=N_samples,S_shifts=S_shifts), achieved_loss.detach().numpy(), trained_nfm


def QMC_fourier_pricing_engine(model_name, payoff_name, option_params, N_samples, S_shifts, transform_distribution=None, transform_params=None, weights=None, TOLR = None):
    
    # Initialize model
    if model_name == 'NIG':
        d, S0, K, r, T, q, alpha, beta, delta, DELTA = option_params
        model = NIGModel(d, T, r, q, alpha, beta, delta, DELTA)
        if transform_distribution is None:
            transform_distribution = 'ML'
            transform_params = 2 / (delta**2 * T**2) * la.inv(DELTA)
        
    elif model_name == 'GH':
        d,S0,K,r,T,q,alpha,beta,delta,DELTA,lambda_var = option_params
        model = GHModel(d, T, r, q, alpha, beta, delta, DELTA, lambda_var)
        if transform_distribution is None:
            transform_distribution = 'ML'
            transform_params = 2 / (delta**2 * T**2) * la.inv(DELTA)
        
    elif model_name == 'VG':
        d,S0,K,r,q,T,SIGMA,theta,nu = option_params
        model = VGModel(d, T, r, q, SIGMA, theta, nu)
        if transform_distribution is None:
            transform_distribution = 'MS'
            transform_params = (la.inv(SIGMA), 2*T / nu - d)
            
    elif model_name == 'GBM':
        d,S0,K,r,T,q,SIGMA = option_params
        model = GBMModel(d, T, r, q, SIGMA)
        if transform_distribution is None:
            transform_distribution = 'MN'
            transform_params = 1 / T * la.inv(SIGMA)
    
    # Initialize payoff
    if payoff_name == 'call_on_min':
        payoff = CallOnMin()
    
    elif payoff_name == 'put_on_max':
        payoff = PutOnMax()
        
    elif payoff_name == 'spread_call':
        payoff = SpreadCall()
   
    elif payoff_name == 'basket_put':
        if weights is None:
            weights = (1/d) * np.ones(d)  # Default to equi-weighted option
        payoff = BasketPut(weights=weights)
    
    elif payoff_name == 'cash_or_nothing_put':
        payoff = CashOrNothingPut()
    
    # Initialize transformation based on distribution type
    if transform_distribution == "ML":
        transform_distribution_instance = MLTransform(transform_params)
    
    elif transform_distribution == "MS":
        SIGMA_IS, nu_IS = transform_params
        transform_distribution_instance = MSTransform(SIGMA_IS=SIGMA_IS, nu_IS=nu_IS)
        
    elif transform_distribution == "MN":
        SIGMA_IS_ = transform_params
        transform_distribution_instance = MNTransform(SIGMA_IS=SIGMA_IS_)
    

    # Create pricing engine
    engine = PricingEngine(model=model, payoff=payoff, transform=transform_distribution_instance)
    
    
    if S_shifts < 30:
        print("Warning: The statistical error estimation of RQMC is not reliable because S_shifts < 30.")
    # Compute price
    if (TOLR is not None): # if the user specifies targer relative error
        rel_stat_error = 2 * TOLR # Initialize relative statistical error
        N_samples = 2
        while (rel_stat_error > TOLR):
            N_samples = 2 * N_samples # double the number of samples if relative tolerance not achieved.
            price, stat_error = engine.compute_price(d=d,S0=S0,K=K,r=r,T=T,N_samples=N_samples,S_shifts=S_shifts)
            rel_stat_error = stat_error / price
        return price, stat_error
    else:
        return engine.compute_price(d=d,S0=S0,K=K,r=r,T=T,N_samples=N_samples,S_shifts=S_shifts)