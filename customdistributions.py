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


class Multivariate_Diag_t(nf.distributions.base.BaseDistribution):
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

    def __init__(self, loc, diag, dfs):  
       
        super().__init__()
        self.loc = loc
        self.diag = diag
        self.df = dfs
        print([(lo, dia,df)for lo, dia, df in zip(loc,diag,dfs)])
        self.distributions = [torch.distributions.studentT.StudentT(df,lo, dia)for lo, dia,df in zip(loc,diag,dfs)]
        self.n_dims = len(diag)
        self.max_log_prob = 0.0

    def log_prob(self, z):
        return sum([distribution.log_prob(z[:,i]) for i,distribution in enumerate(self.distributions)])
    
    def sample(self, num_samples = 1):
        if type(num_samples) is torch.Size:
            return torch.stack([distribution.sample(num_samples) for distribution in self.distributions],axis = 1)
        else:
            return torch.stack([distribution.sample(torch.Size([num_samples])) for distribution in self.distributions],axis = 1)
    
    def forward(self, num_samples = 1, context = None):
        z = self.sample(num_samples)
        return torch.tensor(z), self.log_prob(z)


class Multivariate_t(nf.distributions.target.Target):
    """
    Multivariate student t distribution wrapper from scipy.stats to normflows base distributions class.
    Instantiates scipy.stats.multivariate_t_frozen istribution of the given parameters and makes the 
    scipy distribution calls accessible for normflows (torch) ML functionalities.
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
        res = torch.tensor(self.distribution.logpdf(z.detach().numpy()))
        return res   #Inefficient but should work?