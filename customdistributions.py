import numpy as np
import normflows as nf
import scipy as sc
from scipy import stats
from scipy.stats import sampling
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import abc
import math
import qmcpy
import torch_betainc



def student_t_icdf(p,distribution, initial_guess=None, steps=10):
    # p: probabilities, df: degrees of freedom
    x = initial_guess if initial_guess is not None else torch.zeros_like(p)
    for _ in range(steps):
        cdf = distribution.cdf(x)       #differentiable CDF
        pdf = torch.exp(distribution.log_prob(x))       # differentiable PDF
        # Newton update
        x = x - (cdf - p) / pdf
    return x

class Multivariate_Diag_t_Torch(nf.distributions.base.BaseDistribution):
    """
    Diagonal Multivariate t-student implementation as normflows base. Underlying distributions
    are torch.student_t distributions. log_prob and sample wrap the relevant function calls and stack/sum the results to convert
    to multidimensional case.
    Parameters:
    loc [torch.tensor([torch.float64])]: location of distribution center
    diag [torch.tensor([torch.float64]))]: diagonal entries of the distribution matrix
    df [torch.float64]: number of degrees of freedom of the student t distribution
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    sample(num_samples: int) -> torch.tensor([torch.float64]): returns num_samples samples from the underlying distribution
    forward(num_samples: int = 1) -> Sequence(torch.float64, torch.tensor([torch.float64])):
        returns Sequence of distribution samples and the log_prob of the sampled points
    """

    def __init__(self, loca, diag, df):  
       
        super().__init__()
        self.loc = loca
        self.diag = diag
        self.df = df
        self.distributions = [torch.distributions.studentT.StudentT(df=df,loc=loc, scale=scale)for loc, scale in zip(loca,diag)]
        self.n_dims = len(diag)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        res = torch.tensor(sum([distribution.log_prob(z[:,i]) for i,distribution in enumerate(self.distributions)]))
        return res
    
    def sample(self, num_samples = 1,**kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z
    
    def forward(self, num_samples = 1, context = None):
        if type(num_samples) is torch.Size:
            z = torch.stack([distribution.sample(num_samples) for distribution in self.distributions],axis = 1)
        else:
            z = torch.stack([distribution.sample(torch.Size([num_samples])) for distribution in self.distributions],axis = 1)
        return z, self.log_prob(z)
    

class Multivariate_Diag_Base_MC_mapping(nf.distributions.base.BaseDistribution):
    """
    Diagonal Multivariate t-student implementation as normflows base with qmc sampling. Underlying distributions
    are torch_betainc.student_t distributions and for the inverse cdf we use standard scipy t. 
    log_prob and sample wrap the relevant function calls and stack/sum the results to convert to multidimensional case.
    Parameters:
    loc, [torch.tensor([torch.float64])]: location of distribution center
    diag, [torch.tensor([torch.float64]))]: diagonal entries of the distribution matrix
    df, float: number of degrees of freedom of the student t distribution
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    sample(num_samples: int) -> torch.tensor([torch.float64]): returns num_samples samples from the underlying distribution
    forward(num_samples: int = 1) -> Sequence(torch.float64, torch.tensor([torch.float64])):
        returns Sequence of distribution samples and the log_prob of the sampled points
    """

    def __init__(self, loca, diag, df):  
       
        super().__init__()
        self.loc = loca
        self.diag = diag
        self.df = df
        self.distributions = [torch_betainc.StudentT(df=df,loc=loc, scale=scale)for loc, scale in zip(loca,diag)]
        self.n_dims = len(diag)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        res = sum([distribution.log_prob(z[:,i]) for i,distribution in enumerate(self.distributions)])
        return res
    
    def sample(self, num_samples = 1,**kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z
    
    def forward(self, num_samples = 1, context = None):
        xi_RQMC = torch.rand((num_samples,self.n_dims)) # generates digitally shifted Sobol points.
        z = torch.stack([torch.tensor(scale*stats.t.ppf(np.array(xi_RQMC[:,i]),self.df)+loc) for i,(loc,scale) in enumerate(zip(self.loc,self.diag))],axis = 1)
        #Sanity check:          
        #print(self.distributions[i].cdf(z[])-torch.tensor(xi_RQMC)) 
        #Should give something close 0 for the shape of the sample!   
        return z, self.log_prob(z)

class Multivariate_Diag_t_qmc(nf.distributions.base.BaseDistribution):
    """
    Diagonal Multivariate t-student implementation as normflows base with qmc sampling. Underlying distributions
    are torch_betainc.student_t distributions and for the inverse cdf we use standard scipy t. 
    log_prob and sample wrap the relevant function calls and stack/sum the results to convert to multidimensional case.
    Parameters:
    loc, [torch.tensor([torch.float64])]: location of distribution center
    diag, [torch.tensor([torch.float64]))]: diagonal entries of the distribution matrix
    df, float: number of degrees of freedom of the student t distribution
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    sample(num_samples: int) -> torch.tensor([torch.float64]): returns num_samples samples from the underlying distribution
    forward(num_samples: int = 1) -> Sequence(torch.float64, torch.tensor([torch.float64])):
        returns Sequence of distribution samples and the log_prob of the sampled points
    """

    def __init__(self, loca, diag, df):  
       
        super().__init__()
        self.loc = loca
        self.diag = diag
        self.df = df
        self.distributions = [torch_betainc.StudentT(df=df,loc=loc, scale=scale)for loc, scale in zip(loca,diag)]
        self.n_dims = len(diag)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        res = sum([distribution.log_prob(z[:,i]) for i,distribution in enumerate(self.distributions)])
        return res
    
    def sample(self, num_samples = 1,**kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z
    
    def forward(self, num_samples = 1, context = None):
        xi_RQMC = qmcpy.DigitalNetB2(self.n_dims, order = "GRAY", randomize='LMS DS').gen_samples(num_samples) # generates digitally shifted Sobol points.
        z = torch.stack([torch.tensor(scale*stats.t.ppf(np.array(xi_RQMC[:,i]),self.df)+loc) for i,(loc,scale) in enumerate(zip(self.loc,self.diag))],axis = 1)
        #Sanity check:          
        #print(self.distribution.cdf(z)-torch.tensor(xi_RQMC)) 
        #Should give something close 0 for the shape of the sample!   
        return z, self.log_prob(z)

class Multivariate_Diag_norm_qmc(nf.distributions.base.BaseDistribution):
    """
    Diagonal Multivariate t-student implementation as normflows base. Underlying distributions
    are torch.student_t distributions. log_prob and sample wrap the relevant function calls and stack/sum the results to convert
    to multidimensional case.
    Parameters:
    loc [torch.tensor([torch.float64])]: location of distribution center
    diag [torch.tensor([torch.float64]))]: diagonal entries of the distribution matrix
    df [torch.float64]: number of degrees of freedom of the student t distribution
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    sample(num_samples: int) -> torch.tensor([torch.float64]): returns num_samples samples from the underlying distribution
    forward(num_samples: int = 1) -> Sequence(torch.float64, torch.tensor([torch.float64])):
        returns Sequence of distribution samples and the log_prob of the sampled points
    """

    def __init__(self, loca, diag):  
       
        super().__init__()
        self.loc = loca
        self.diag = diag
        self.distributions = [torch.distributions.Normal(loc=loc, scale=scale)for loc, scale in zip(loca,diag)]
        self.n_dims = len(diag)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        res = sum([distribution.log_prob(z[:,i]) for i,distribution in enumerate(self.distributions)])
        return res
    
    def sample(self, num_samples = 1,**kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z
    
    def forward(self, num_samples = 1, context = None):
        xi_RQMC = torch.tensor(qmcpy.DigitalNetB2(self.n_dims, order = "GRAY", randomize='LMS DS').gen_samples(num_samples)) # generates digitally shifted Sobol points.
        #print(xi_RQMC.shape)
        z = torch.stack([distribution.icdf(xi_RQMC[:,i]) for i,distribution in enumerate(self.distributions)],axis = 1)
        #print(z.shape)
        return z, self.log_prob(z)


class MultivariateStudentT(nf.distributions.target.Target):
    def __init__(self, loc, cov, df):
        """
        Multivariate Student-t with full covariance matrix.

        Args:
            loc (torch.tensor): mean vector of shape [n_dims]
            cov (torch.tensor): covariance matrix of shape [n_dims, n_dims], positive definite
            df (float): degrees of freedom
        """
        super().__init__()
        self.loc = torch.tensor(loc, dtype=torch.float64)
        self.cov = torch.tensor(cov, dtype=torch.float64)
        self.df = df
        self.n_dims = self.loc.shape[0]
        self.L = torch.linalg.cholesky(self.cov).type(torch.DoubleTensor)  # Cholesky factor for sampling

    def sample(self, num_samples=1):
        # Sample chi-squared
        w = torch.distributions.Chi2(self.df).sample((num_samples, 1)).type(torch.DoubleTensor)
        # Sample standard normal
        z = torch.randn((num_samples, self.n_dims)).type(torch.DoubleTensor)
        # Apply transformation
        x = self.loc + (z @ self.L.T) / torch.sqrt(w / self.df)
        return x

    def log_prob(self, x):
        """
        Compute log pdf of multivariate t distribution.
        Formula: 
        log p(x) = log Gamma((ν + d)/2) - log Gamma(ν/2) - 0.5 log |Σ| - d/2 log(νπ)
                    - (ν + d)/2 * log(1 + Mahalanobis / ν)
        """
        d = torch.tensor(self.n_dims, dtype=x.dtype, device=x.device)
        x_mu = x - self.loc
        # Mahalanobis distance
        L_inv = torch.linalg.inv(self.L)
        y = x_mu @ L_inv.T
        mahal = torch.sum(y**2, dim=-1)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(self.L)))
        from torch import lgamma, log, pi
        nu = torch.tensor(self.df, dtype=x.dtype, device=x.device)
        pi = torch.tensor(torch.pi, dtype=x.dtype, device=x.device)
        log_norm = lgamma((nu + d) / 2) - lgamma(nu / 2) - 0.5 * log_det - (d / 2) * log(nu * pi)
        log_prob = log_norm - ((nu + d)/2) * torch.log(1 + mahal / nu)
        return log_prob

    def forward(self, num_samples=1, context=None):
        x = self.sample(num_samples)
        return x, self.log_prob(x)
