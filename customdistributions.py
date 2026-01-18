import numpy as np
import normflows as nf
import scipy as sc
from scipy import stats
from scipy.stats import sampling
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import abc


class Multivariate_Diag_t(nf.distributions.base.BaseDistribution):
    """
    Multivariate student t distribution wrapper from scipy.stats to normflows base distributions class.
    Instantiates scipy.stats.multivariate_t_frozen istribution of the given parameters and makes the 
    scipy distribution calls accessible for normflows (torch) ML functionalities.
    Only allows Diagonal matrix entries!
    Parameters:
    loc [torch.tensor([torch.float64])]: location of distribution center
    diag [torch.tensor([torch.float64]))]: diagonal entries of the distribution matrix
    df [torch.float64]: number of degrees of freedom of the student t distribution
    Optional Parameters:
    sampler_reolution [float], default 1e-10: resolution of the sampler. 1e-10 is the resolution limit for numpy random state
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    sample(num_samples: int) -> torch.tensor([torch.float64]): returns num_samples samples from the underlying distribution
    forward(num_samples: int = 1) -> Sequence(torch.float64, torch.tensor([torch.float64])):
        returns Sequence of distribution samples and the log_prob of the sampled points
    """

    def __init__(self, loc, diag, df, sampler_resolution = 1e-10):  
       
        super().__init__()
        self.loc = loc
        self.diag = diag
        self.df = df
        self.distribution = stats.multivariate_t(loc, torch.diag(diag),df)
        self.rng = np.random.default_rng()
        self.sampler = sampling.NumericalInverseHermite(self.distribution,u_resolution = sampler_resolution)

    def log_prob(self, z):
        return self.distribution.logpdf(z)
    
    def sample(self, num_samples = 1):
        return self.sampler.rvs(size = num_samples)     #Scipy also allows qrvc for qmc sampling, maybe intersting for performance?
    
    def forward(self, num_samples = 1, context = None):
        z = self.sample(num_samples)
        return z, self.log_prob(z)


class Multivariate_t(nf.distributions.target.Target):
    """
    Multivariate student t distribution wrapper from scipy.stats to normflows base distributions class.
    Instantiates scipy.stats.multivariate_t_frozen istribution of the given parameters and makes the 
    scipy distribution calls accessible for normflows (torch) ML functionalities.
    Only allows Diagonal matrix entries!
    Parameters:
    loc [torch.tensor([torch.float64])]: location of distribution center
    matrix [torch.tensor([torch.float64],[torch.float64]))]: distribution matrix
    df [torch.float64]: number of degrees of freedom of the student t distribution
    Methods:
    log_prob(z: torch.tensor([torch.float64])) -> torch.float64: returns log probability of sample point z
    """

    def __init__(self, loc, matrix, df):
       
        super().__init__()
        self.loc = loc
        self.matrix = matrix
        self.df = df
        self.distribution = stats.multivariate_t(loc, matrix,df)
        self.n_dims = len(matrix)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        return torch.tensor(self.distribution.logpdf(z))